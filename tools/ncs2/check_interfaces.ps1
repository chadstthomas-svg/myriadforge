# Read-only check: is a WinUSB device interface registered for the NCS2
# under the GUIDs the OpenVINO MYRIAD plugin enumerates
# (XLink win_usb.c: GUID_DEVINTERFACE_Myriad2 = 19E08104-...,
#  GUID_DEVINTERFACE_MyriadX = 504E1220-...)?
$guids = @('{19E08104-0543-40A5-B107-87EF463DCEF1}', '{504E1220-E189-413A-BDEC-ECFFAF3A3731}')
foreach ($g in $guids) {
    $p = "HKLM:\SYSTEM\CurrentControlSet\Control\DeviceClasses\$g"
    if (-not (Test-Path $p)) { Write-Host "$g : class key absent"; continue }
    $children = Get-ChildItem $p
    $match = $children | Where-Object { $_.PSChildName -like '*VID_03E7*' }
    if ($match) {
        Write-Host "$g : NCS2 interface REGISTERED -> $($match.PSChildName)"
    } else {
        Write-Host "$g : class key present, no NCS2 instance ($($children.Count) other devices)"
    }
}
# Also show which interface classes the device instance itself exposes.
$dp = "HKLM:\SYSTEM\CurrentControlSet\Enum\USB\VID_03E7&PID_2150\03E72150\Device Parameters"
if (Test-Path $dp) {
    Get-ItemProperty $dp | Select-Object DeviceInterfaceGUID, DeviceInterfaceGUIDs | Format-List
}
