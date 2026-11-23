"use strict";
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
const RELEASE_DATA = "https://huggingface.co/spaces/racklab/BankShot-Search/resolve/b5bd82b5ab3a77f2eacfdec06ecac7d55f023b76/data/";
importScripts(ORT_CDN + "ort.wasm.min.js");
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = ORT_CDN;

const domains = new Map();
let model;
const norm = (value) => (String(value).toLowerCase().match(/[a-z0-9]+/g) || []).join(" ");
const stop = new Set("a an and are as at be by can do does for from how i in is it my of on or our that the their this to was we what where which will with you".split(" "));
const terms = (text) => norm(text).split(" ").filter(word => word.length > 1 && !stop.has(word));
async function asset(name) {
  const response = await fetch(RELEASE_DATA + name);
  if (!response.ok) throw Error("A search asset could not load. Try again or open Field Notes.");
  return response;
}
function rowsFor(domain) {
  if (!domains.has(domain)) {
    const rows = asset(domain + ".json").then(response => response.json());
    domains.set(domain, rows);
    rows.catch(() => domains.delete(domain));
  }
  return domains.get(domain);
}
function ready() {
  if (!model) model = Promise.all([
    asset("vocab.json").then(r => r.json()),
    asset("tokens.f32").then(r => r.arrayBuffer()).then(b => new Float32Array(b)),
    asset("scorer.onnx").then(r => r.arrayBuffer()).then(bytes =>
      ort.InferenceSession.create(bytes,{executionProviders:["wasm"]})),
  ]).catch(error => {model=undefined; throw error;});
  return model;
}
function words(text,vocab) {
  const clean = text.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu,"")
    .replace(/\s|\p{Z}/gu," ").replace(/[\p{Cc}\p{Cf}]/gu,"")
    .replace(/[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e\p{P}]/gu," $& ");
  const output = [vocab["[CLS]"]];
  for (const term of clean.split(/\s+/u).filter(Boolean)) {
    const parts = [];
    let start = 0;
    while (start < term.length) {
      let end = term.length, id;
      for (;end>start;end--) {
        const candidate = (start ? "##" : "") + term.slice(start,end);
        if (Object.hasOwn(vocab,candidate)) {id=vocab[candidate];break;}
      }
      if (id === undefined) {parts.splice(0,parts.length,vocab["[UNK]"]);break;}
      parts.push(id);
      start=end;
    }
    output.push(...parts.slice(0,47-output.length));
    if (output.length === 47) break;
  }
  output.push(vocab["[SEP]"]);
  return output;
}
async function semantic(text,scope) {
  const [vocab,vectors,session] = await ready();
  const ids = words(text,vocab);
  const field = new Float32Array(64*48), mask = new Float32Array(48);
  mask.fill(1,0,ids.length);
  for (let t=0;t<ids.length;t++) for (let c=0;c<64;c++) field[c*48+t]=vectors[ids[t]*64+c];
  const scores=(await session.run({token_vectors:new ort.Tensor("float32",field,[1,64,1,48]),
    token_mask:new ort.Tensor("float32",mask,[1,1,1,48])})).scores.data;
  if (scores.length!==10000 || !scores.every(Number.isFinite)) throw Error("Browser scorer returned invalid data");
  return scope.slice().sort((a,b)=>scores[b.i]-scores[a.i] || a.i-b.i).slice(0,100);
}
function count(tokens) {
  const bag=new Map();
  for(const token of tokens) bag.set(token,(bag.get(token)||0)+1);
  return bag;
}
function lexical(text,scope) {
  const query=[...new Set(terms(text))].slice(0,192);
  if (!query.length) return [];
  for(const row of scope) if(!row.index) {
    const body=terms(row.text);
    row.index={body:count(body),title:count(terms(row.title+" "+row.heading)),length:body.length};
  }
  const average=Math.max(1,scope.reduce((sum,row)=>sum+row.index.length,0)/scope.length);
  const weights=query.map(term=>{
    let df=0;
    for(const row of scope) if(row.index.body.has(term)||row.index.title.has(term)) df++;
    return Math.log1p((scope.length-df+.5)/(df+.5));
  });
  return scope.map(row=>{
    const length=1.2*(.25+.75*row.index.length/average);
    let score=0;
    for(let i=0;i<query.length;i++){
      const tf=(row.index.body.get(query[i])||0)+2*(row.index.title.get(query[i])||0);
      score+=weights[i]*2.2*tf/(tf+length);
    }
    return {row,score};
  }).filter(item=>item.score>0).sort((a,b)=>b.score-a.score||a.row.i-b.row.i).slice(0,100).map(item=>item.row);
}
async function search({text,domain,vehicle}) {
  const started=performance.now();
  if (!["dealership","manufacturing","store_security"].includes(domain)) throw Error("Unknown catalog");
  const rows=await rowsFor(domain);
  const scope=vehicle && domain==="dealership" ? rows.filter(row=>row.vehicles.some(([make,name,year])=>
    Number(year)===vehicle[0] && norm(make)===norm(vehicle[1]) && norm(name)===norm(vehicle[2]))) : rows;
  function complete(status,path,ranked=[]) {
    const hits=new Map();
    for(const {id,section_id,title,url} of ranked){
      if(!hits.has(id)) hits.set(id,{id,section_id,title,url});
      if(hits.size===10) break;
    }
    return {status,path,hits:[...hits.values()],request_ms:performance.now()-started,backend:"browser_cpu"};
  }
  if(!scope.length) return complete("no_match","vehicle");
  const match=domain==="manufacturing" ? text.match(/\b1910\.\d{1,4}\b/g)
    : domain==="dealership" ? text.toUpperCase().match(/\b\d{2}[A-Z]\d{6}\b/g) : null;
  const ids=[...new Set(match||[])];
  if(ids.length>1) return complete("ambiguous","identifier");
  if(ids.length===1){
    const target=scope.find(row=>row.id.split("/")[1]===ids[0]);
    return target ? complete("found","identifier",[target]) : complete("no_match","identifier");
  }
  if(domain==="store_security") {
    const path=text.trim().replace(/^https?:\/\/docs\.frigate\.video\//i,"").split(/[?#]/u)[0].replace(/^\/+|\/+$/g,"");
    if(/^[a-z0-9_-]+(?:\/[a-z0-9_-]+)+$/u.test(path)) {
      const target=scope.find(row=>row.id===path);
      if(target) return complete("found","guide_reference",[target]);
    }
  }
  if(!norm(text)) return vehicle ? complete("found","vehicle",scope) : complete("needs_details","catalog");
  const sparse=lexical(text,scope), dense=await semantic(text,scope);
  const votes=new Map();
  for(const [lane,weight] of [[sparse,1],[dense,.8]]) {
    lane.forEach((row,index)=>votes.set(row,(votes.get(row)||0)+weight/(41+index)));
  }
  const ranked=[...votes.keys()].sort((a,b)=>votes.get(b)-votes.get(a)||a.i-b.i);
  return complete(ranked.length?"found":"no_match","browser_hybrid",ranked);
}
self.addEventListener("message",async ({data})=>{
  try {self.postMessage({result:await search(data)});}
  catch(error){self.postMessage({error:String(error.message||error)});}
});
