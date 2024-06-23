<powershell>

function Show-SecurityProtocol {
    $currentProtocol = [System.Net.ServicePointManager]::SecurityProtocol
    Write-Host "Current SecurityProtocol: $currentProtocol"
    
    # Display individual protocols enabled
    $protocolNames = @()
    if ($currentProtocol -band [System.Net.SecurityProtocolType]::Ssl3) { $protocolNames += "SSL3" }
    if ($currentProtocol -band [System.Net.SecurityProtocolType]::Tls) { $protocolNames += "TLS1.0" }
    if ($currentProtocol -band [System.Net.SecurityProtocolType]::Tls11) { $protocolNames += "TLS1.1" }
    if ($currentProtocol -band [System.Net.SecurityProtocolType]::Tls12) { $protocolNames += "TLS1.2" }
    if ($currentProtocol -band [System.Net.SecurityProtocolType]::Tls13) { $protocolNames += "TLS1.3" }
    
    Write-Host "Enabled protocols: $($protocolNames -join ', ')"
}

Get-Date | Out-File -FilePath "C:\Users\Administrator\STARTED"


Write-Host "Listing Execution policies..."
Get-ExecutionPolicy
Get-ExecutionPolicy -List

# Check SecurityProtocol before change.
Write-Host "Before change:"
Show-SecurityProtocol

# Enable TLSv1.2 and TLSv1.3.
# See https://learn.microsoft.com/en-us/dotnet/api/system.net.securityprotocoltype?view=net-8.0
Write-Host "`nSetting SecurityProtocol..."
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072 -bor 12288;

# Disable TLS v1.0 and v1.1. 
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -band (-bnot 192) -band (-bnot 768);

# Check SecurityProtocol after change.
Write-Host "`nAfter change:"
Show-SecurityProtocol



# Install Chocolatey package manager.
# See https://chocolatey.org/install
Write-Host "Installing Chocolatey..."
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"


Write-Host "Installing visual studio..."

# Install Visual Studio 2022 Community Edition
choco install visualstudio2022community -y

# Install necessary workloads and components
choco install visualstudio2022-workload-nativedesktop -y  # C++ desktop development
choco install visualstudio2022-workload-nativegame -y     # Game development with C++
choco install windows-sdk-10-version-2004-all -y          # Windows 10 SDK



Get-Date | Out-File -FilePath "C:\Users\Administrator\COMPLETED"

</powershell>