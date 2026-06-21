; Volterra NL Filter — NSIS Installer
; Installs: VST3 plugin + Standalone app
; Targets 64-bit Windows only.

Unicode True

!define PRODUCT_NAME      "Volterra NL Filter"
!define PRODUCT_VERSION   "1.0.0"
!define COMPANY_NAME      "VoltDSP"
!define EXE_NAME          "Volterra NL Filter.exe"
!define VST3_DIR_NAME     "Volterra NL Filter.vst3"
!define UNINSTALL_KEY     "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

; Build artifacts (relative to this .nsi file)
!define BUILD_RELEASE     "build\VolterraPlugin_artefacts\Release"
!define VST3_SRC          "${BUILD_RELEASE}\VST3\${VST3_DIR_NAME}"
!define STANDALONE_SRC    "${BUILD_RELEASE}\Standalone\${EXE_NAME}"

; Install destinations
!define VST3_DEST         "$COMMONFILES64\VST3"
!define APP_DEST          "$PROGRAMFILES64\${COMPANY_NAME}\${PRODUCT_NAME}"

; ─── General ──────────────────────────────────────────────────────────────────
Name              "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile           "build\Volterra-NL-Filter-${PRODUCT_VERSION}-win64-Setup.exe"
InstallDir        "${APP_DEST}"
InstallDirRegKey  HKLM "${UNINSTALL_KEY}" "InstallLocation"
RequestExecutionLevel admin
SetCompressor     /SOLID lzma
SetCompress       auto

; ─── MUI2 UI ─────────────────────────────────────────────────────────────────
!include "MUI2.nsh"

!define MUI_ABORTWARNING
!define MUI_ICON   "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ─── Version info embedded in .exe ───────────────────────────────────────────
VIProductVersion  "1.0.0.0"
VIAddVersionKey   "ProductName"      "${PRODUCT_NAME}"
VIAddVersionKey   "CompanyName"      "${COMPANY_NAME}"
VIAddVersionKey   "FileVersion"      "${PRODUCT_VERSION}"
VIAddVersionKey   "ProductVersion"   "${PRODUCT_VERSION}"
VIAddVersionKey   "FileDescription"  "${PRODUCT_NAME} Installer"
VIAddVersionKey   "LegalCopyright"   "© 2026 ${COMPANY_NAME}"

; ─── Sections ─────────────────────────────────────────────────────────────────

Section "VST3 Plugin" SEC_VST3
    SectionIn RO          ; required — cannot be deselected

    SetOutPath "${VST3_DEST}\${VST3_DIR_NAME}\Contents\x86_64-win"
    File "${VST3_SRC}\Contents\x86_64-win\${VST3_DIR_NAME}"

    SetOutPath "${VST3_DEST}\${VST3_DIR_NAME}\Contents\Resources"
    File "${VST3_SRC}\Contents\Resources\moduleinfo.json"

    ; Write uninstall info after installing the first component
    WriteRegStr   HKLM "${UNINSTALL_KEY}" "DisplayName"     "${PRODUCT_NAME}"
    WriteRegStr   HKLM "${UNINSTALL_KEY}" "DisplayVersion"  "${PRODUCT_VERSION}"
    WriteRegStr   HKLM "${UNINSTALL_KEY}" "Publisher"       "${COMPANY_NAME}"
    WriteRegStr   HKLM "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr   HKLM "${UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
    WriteRegDWORD HKLM "${UNINSTALL_KEY}" "NoModify"        1
    WriteRegDWORD HKLM "${UNINSTALL_KEY}" "NoRepair"        1

    ; Uninstaller
    SetOutPath "$INSTDIR"
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Standalone Application" SEC_STANDALONE
    SetOutPath "$INSTDIR"
    File "${STANDALONE_SRC}"

    ; Start Menu shortcut
    CreateDirectory "$SMPROGRAMS\${COMPANY_NAME}"
    CreateShortcut  "$SMPROGRAMS\${COMPANY_NAME}\${PRODUCT_NAME}.lnk" \
                    "$INSTDIR\${EXE_NAME}"
SectionEnd

; ─── Section descriptions (shown in Components page) ─────────────────────────
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SEC_VST3}       \
        "64-bit VST3 plugin — installs to $COMMONFILES64\VST3"
    !insertmacro MUI_DESCRIPTION_TEXT ${SEC_STANDALONE} \
        "Standalone application — no host required"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ─── Uninstaller ──────────────────────────────────────────────────────────────
Section "Uninstall"
    ; Remove VST3
    RMDir /r "${VST3_DEST}\${VST3_DIR_NAME}"

    ; Remove standalone app and Start Menu shortcut
    Delete "$INSTDIR\${EXE_NAME}"
    Delete "$SMPROGRAMS\${COMPANY_NAME}\${PRODUCT_NAME}.lnk"
    RMDir  "$SMPROGRAMS\${COMPANY_NAME}"

    ; Remove uninstaller and install directory
    Delete "$INSTDIR\Uninstall.exe"
    RMDir  "$INSTDIR"
    RMDir  "$PROGRAMFILES64\${COMPANY_NAME}"

    DeleteRegKey HKLM "${UNINSTALL_KEY}"
SectionEnd
