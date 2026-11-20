param(
    [Parameter(Mandatory)][string]$Plan   # JSONL: {"id","text","voice","rate","pitch","out"}
)
# Batch SAPI synthesis for the voice-to-tool fixtures: one process, one WAV per
# plan line, forced 16 kHz 16-bit mono. Pitch moves go through SSML prosody
# (the SpeechSynthesizer has no pitch property); rate uses the Rate property.
# SetOutputToNull() after each Speak closes the wave file without playing audio.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

$fmt = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
    16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()

$done = 0
Get-Content $Plan | ForEach-Object {
    if (-not $_.Trim()) { return }
    $item = $_ | ConvertFrom-Json
    $synth.SelectVoice([string]$item.voice)
    $synth.Rate = [int]$item.rate
    $pitch = [string]$item.pitch
    if ($pitch -eq '0%') {
        $synth.SetOutputToWaveFile([string]$item.out, $fmt)
        $synth.Speak([string]$item.text)
        $synth.SetOutputToNull()
    } else {
        $ssml = '<speak version="1.0" xml:lang="en-US"><prosody pitch="' + $pitch + '">' + `
                [System.Security.SecurityElement]::Escape([string]$item.text) + '</prosody></speak>'
        $synth.SetOutputToWaveFile([string]$item.out, $fmt)
        $synth.SpeakSsml($ssml)
        $synth.SetOutputToNull()
    }
    $done++
    if ($done % 200 -eq 0) { Write-Output ("rendered " + $done) }
}
Write-Output ("done " + $done)
