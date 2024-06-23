#<powershell>

# Function to display the current SecurityProtocol
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

Get-ExecutionPolicy -List

# Set execution policy to allow script execution.
# Other options are: AllSigned, RemoteSigned.
# see
# https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy?view=powershell-7.4
Set-ExecutionPolicy Bypass -Scope Process -Force

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


# Backup existing Chocolatey folder.
if (Test-Path "C:\ProgramData\chocolatey") {
    Write-Host "Backing up existing Chocolatey installation..."
    Copy-Item "C:\ProgramData\chocolatey" "C:\ProgramData\chocolatey_backup" -Recurse
    Write-Host "Backup created at C:\ProgramData\chocolatey_backup"
}

# Remove existing Chocolatey folder.
if (Test-Path "C:\ProgramData\chocolatey") {
    Write-Host "Removing existing Chocolatey folder..."
    Remove-Item "C:\ProgramData\chocolatey" -Recurse -Force
    Write-Host "Existing Chocolatey folder removed."
}


# Install Chocolatey package manager.
# See https://chocolatey.org/install
Write-Host "Installing Chocolatey..."
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

# Create a new WebClient object.
$webClient = New-Object System.Net.WebClient
# Download the install script as a string.
$installScript = $webClient.DownloadString('https://community.chocolatey.org/install.ps1')
# Execute the downloaded script.
Invoke-Expression $installScript

# Verify installation.
if (Test-Path "C:\ProgramData\chocolatey\bin\choco.exe") {
    Write-Host "Chocolatey has been successfully installed!"
} else {
    Write-Host "Chocolatey installation might have failed. Please check for any error messages above."
}

# Cleanup.
if (Test-Path "C:\ProgramData\chocolatey_backup") {
    Write-Host "Installation complete. You can now delete the backup at C:\ProgramData\chocolatey_backup if everything is working correctly."
}

# Refresh environment variables.
# This is actually a batch file.
refreshenv


# It reads the environment variables from both the system (HKLM) and user (HKCU) registry locations.
# It updates the current PowerShell session's environment variables with these values.
# It specifically handles the PATH variable, combining both system and user PATHs.
function Refresh-Environment {
    $locations = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
                 'HKCU:\Environment'

    $locations | ForEach-Object {
        $k = Get-Item $_
        $k.GetValueNames() | ForEach-Object {
            $name  = $_
            $value = $k.GetValue($_)
            Set-Item -Path Env:\$name -Value $value
        }
    }

    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

Write-Host "Refresh-Environment function has been defined. You can now use it by typing 'Refresh-Environment'."

# Check if you have a profile: Test-Path $PROFILE
# To create a profile: New-Item -Path $PROFILE -Type File -Force
# To open a text editor: notepad $PROFILE
# To use it: Refresh-Environment



# Install Visual Studio 2022 Community Edition
choco install visualstudio2022community -y

# Install necessary workloads and components
choco install visualstudio2022-workload-nativedesktop -y  # C++ desktop development
choco install visualstudio2022-workload-nativegame -y     # Game development with C++
choco install windows-sdk-10-version-2004-all -y          # Windows 10 SDK



Get-Date | Out-File -FilePath "C:\Users\Administrator\COMPLETE"
Get-Content .\COMPLETE


# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Function to find the Visual Studio installation path
function Get-VisualStudioPath {
    $programFiles = ${env:ProgramFiles(x86)}, ${env:ProgramFiles} | Where-Object { $_ }
    foreach ($root in $programFiles) {
        $vsPath = Join-Path $root "Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe"
        if (Test-Path $vsPath) {
            return $vsPath
        }
    }
    return $null
}

# Start Visual Studio
$vsPath = Get-VisualStudioPath
if ($vsPath) {
    Write-Host "Starting Visual Studio..."
    Start-Process $vsPath
} else {
    Write-Host "Visual Studio installation not found. Please check your installation."
}

Write-Host "Installation complete. Here's what was installed:"
Write-Host "1. Visual Studio 2022 Community Edition"
Write-Host "2. C++ desktop development workload (includes MSVC)"
Write-Host "3. Game development with C++ workload (includes DirectX development tools)"
Write-Host "4. Windows 10 SDK"
Write-Host "`nYou're now set up for C++ and DirectX/Direct3D development on Windows!"




# Create a test C++ program
$testProgram = @"
#include <iostream>
#include <windows.h>

int main()
{
    std::cout << "Hello from Visual Studio C++!" << std::endl;
    
    // Test Windows-specific functionality
    SYSTEMTIME st;
    GetSystemTime(&st);
    std::cout << "Current UTC time: " 
              << st.wYear << "-" 
              << st.wMonth << "-" 
              << st.wDay << " " 
              << st.wHour << ":" 
              << st.wMinute << ":" 
              << st.wSecond << std::endl;

    return 0;
}
"@

# Save the test program
$desktopPath = [Environment]::GetFolderPath("Desktop")
$testFilePath = Join-Path $desktopPath "TestProgram.cpp"
$testProgram | Out-File -FilePath $testFilePath -Encoding utf8

Write-Host "Test program created at: $testFilePath"

# Function to find Visual Studio installation
function Get-VisualStudioPath {
    $programFiles = ${env:ProgramFiles(x86)}, ${env:ProgramFiles} | Where-Object { $_ }
    foreach ($root in $programFiles) {
        $vsPath = Join-Path $root "Microsoft Visual Studio\2022\Community"
        if (Test-Path $vsPath) {
            return $vsPath
        }
    }
    return $null
}

# Find Visual Studio Command Prompt
$vsPath = Get-VisualStudioPath
if ($vsPath) {
    $vcvarsallPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (Test-Path $vcvarsallPath) {
        # Create a batch file to compile and run the program
        $batchContent = @"
@echo off
call "$vcvarsallPath" x64
cl.exe /EHsc "$testFilePath"
if %ERRORLEVEL% EQU 0 (
    TestProgram.exe
) else (
    echo Compilation failed.
)
pause
"@
        $batchFilePath = Join-Path $desktopPath "CompileAndRun.bat"
        $batchContent | Out-File -FilePath $batchFilePath -Encoding ascii

        Write-Host "Batch file created at: $batchFilePath"
        Write-Host "Double-click this batch file to compile and run the test program."
    } else {
        Write-Host "Visual Studio Command Prompt not found. Please ensure Visual Studio is installed correctly."
    }
} else {
    Write-Host "Visual Studio installation not found. Please check your installation."
}







# Install Visual Studio Code
choco install vscode -y

# Install MinGW-w64 (GCC for Windows)
choco install mingw -y

# Add MinGW to PATH
$env:Path += ";C:\ProgramData\chocolatey\lib\mingw\tools\install\mingw64\bin"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)

# Install C/C++ extension for VSCode
code --install-extension ms-vscode.cpptools

# Create a simple C++ program to test the setup
$testProgram = @"
#include <iostream>

int main() {
    std::cout << "Hello from EC2!" << std::endl;
    return 0;
}
"@

New-Item -Path "C:\Users\Administrator\Desktop\test.cpp" -ItemType File -Value $testProgram

# Create a simple batch file to compile and run the test program
$batchFile = @"
@echo off
g++ C:\Users\Administrator\Desktop\test.cpp -o C:\Users\Administrator\Desktop\test.exe
C:\Users\Administrator\Desktop\test.exe
pause
"@

New-Item -Path "C:\Users\Administrator\Desktop\compile_and_run.bat" -ItemType File -Value $batchFile

Write-Host "Setup complete. You can now use VSCode and g++ to compile C++ programs."


#</powershell>