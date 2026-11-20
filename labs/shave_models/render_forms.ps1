param([Parameter(Mandatory)][string]$Output, [int]$Count = 720)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$folder = [IO.Directory]::CreateDirectory($Output).FullName
$rng = [Random]::new(471)
$phrases = @('Library','Display','Archive','Filter','Choose a topic','Project','Options','Details','Layout','Preview','Notes')
$roles = @('button','textbox','checkbox','radio')
$rows = [Collections.Generic.List[object]]::new()
for ($index = 0; $index -lt $Count; $index++) {
    $panel = [Windows.Forms.Panel]::new()
    $panel.Size = [Drawing.Size]::new(384,256)
    $tone = $rng.Next(210,255)
    $panel.BackColor = [Drawing.Color]::FromArgb($tone, $tone, [Math]::Min(255,$tone+$rng.Next(0,8)))
    $slots = [Collections.Generic.List[int]]::new()
    0..7 | ForEach-Object { $slots.Add($_) }
    $targets = @()
    for ($j = 0; $j -lt 8; $j++) {
        $p = $rng.Next($slots.Count)
        $slot = $slots[$p]
        $slots.RemoveAt($p)
        $role = if ($j -lt 4) { $roles[$j] } else { 'label' }
        $control = switch ($role) {
            'button'   { [Windows.Forms.Button]::new() }
            'textbox'  { [Windows.Forms.TextBox]::new() }
            'checkbox' { [Windows.Forms.CheckBox]::new() }
            'radio'    { [Windows.Forms.RadioButton]::new() }
            'label'    { [Windows.Forms.Label]::new() }
        }
        $control.Text = $phrases[$rng.Next($phrases.Count)]
        $control.AutoSize = $false
        $x = ($slot % 2) * 184 + $rng.Next(5,37)
        $y = [int][Math]::Floor($slot/2)*62 + $rng.Next(5,31)
        $control.Bounds = [Drawing.Rectangle]::new($x,$y,$rng.Next(84,141),$rng.Next(21,31))
        if ($role -in 'checkbox','radio') { $control.Checked = $rng.Next(2) -eq 1 }
        $panel.Controls.Add($control)
        if ($role -ne 'label') {
            $w = $control.Width
            if ($role -in 'checkbox','radio') { $w = 15 }
            $targets += ,@($x,$y,$w,$control.Height)
        }
    }
    $bitmap = [Drawing.Bitmap]::new(384,256)
    try {
        $panel.DrawToBitmap($bitmap,[Drawing.Rectangle]::new(0,0,384,256))
        # Native EDIT controls skip WM_PRINT in an offscreen parent. Draw the
        # text field explicitly with the Windows Forms border and text renderer.
        $g = [Drawing.Graphics]::FromImage($bitmap)
        try {
            foreach ($field in $panel.Controls) {
                if ($field -is [Windows.Forms.TextBox]) {
                    $bounds = $field.Bounds
                    $g.FillRectangle([Drawing.Brushes]::White,$bounds)
                    [Windows.Forms.ControlPaint]::DrawBorder3D($g,$bounds,[Windows.Forms.Border3DStyle]::Sunken)
                    $bounds.Inflate(-3,-2)
                    [Windows.Forms.TextRenderer]::DrawText($g,$field.Text,$field.Font,$bounds,$field.ForeColor)
                }
            }
        } finally { $g.Dispose() }
        $file = "form-$index.png"
        $bitmap.Save([IO.Path]::Combine($folder,$file),[Drawing.Imaging.ImageFormat]::Png)
        $rows.Add(@{file=$file; boxes=$targets})
    } finally { $bitmap.Dispose(); $panel.Dispose() }
}
$rows | ConvertTo-Json -Depth 5 -Compress | Set-Content -Encoding utf8 (Join-Path $folder 'forms.json')
