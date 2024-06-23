#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <directxmath.h>
#include <directxcolors.h>

#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "dxgi.lib")
#pragma comment (lib, "d3dcompiler.lib")

using namespace DirectX;

// Global Variables
IDXGISwapChain* swapchain;             // the pointer to the swap chain interface
ID3D11Device* dev;                     // the pointer to our Direct3D device interface
ID3D11DeviceContext* devcon;           // the pointer to our Direct3D device context
ID3D11RenderTargetView* backbuffer;    // the pointer to our back buffer

// Function Prototypes
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
void InitD3D(HWND hWnd);     // sets up and initializes Direct3D
void CleanD3D(void);         // closes Direct3D and releases memory
void RenderFrame(void);      // renders a single frame

// The entry point for any Windows program
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    HWND hWnd;
    WNDCLASSEX wc;

    ZeroMemory(&wc, sizeof(WNDCLASSEX));

    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = L"WindowClass";

    RegisterClassEx(&wc);

    RECT wr = { 0, 0, 800, 600 };
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

    hWnd = CreateWindowEx(NULL,
        L"WindowClass",
        L"DirectX Test",
        WS_OVERLAPPEDWINDOW,
        300,
        300,
        wr.right - wr.left,
        wr.bottom - wr.top,
        NULL,
        NULL,
        hInstance,
        NULL);

    ShowWindow(hWnd, nCmdShow);

    InitD3D(hWnd);

    MSG msg;

    while (TRUE)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if (msg.message == WM_QUIT)
                break;
        }

        RenderFrame();
    }

    CleanD3D();

    return msg.wParam;
}

// This is the main message handler for the program
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_DESTROY:
    {
        PostQuitMessage(0);
        return 0;
    } break;
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

void InitD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC scd;

    ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));

    scd.BufferCount = 1;                                   // one back buffer
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;    // use 32-bit color
    scd.BufferDesc.Width = 800;                            // set the back buffer width
    scd.BufferDesc.Height = 600;                           // set the back buffer height
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;     // how swap chain is to be used
    scd.OutputWindow = hWnd;                               // the window to be used
    scd.SampleDesc.Count = 4;                              // how many multisamples
    scd.Windowed = TRUE;                                   // windowed/full-screen mode
    scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;    // allow full-screen switching

    D3D11CreateDeviceAndSwapChain(NULL,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        NULL,
        NULL,
        NULL,
        D3D11_SDK_VERSION,
        &scd,
        &swapchain,
        &dev,
        NULL,
        &devcon);

    ID3D11Texture2D* pBackBuffer;
    swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);

    dev->CreateRenderTargetView(pBackBuffer, NULL, &backbuffer);
    pBackBuffer->Release();

    devcon->OMSetRenderTargets(1, &backbuffer, NULL);

    D3D11_VIEWPORT viewport;
    ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = 800;
    viewport.Height = 600;

    devcon->RSSetViewports(1, &viewport);
}

void CleanD3D()
{
    swapchain->SetFullscreenState(FALSE, NULL);    // switch to windowed mode

    // close and release all existing COM objects
    backbuffer->Release();
    swapchain->Release();
    dev->Release();
    devcon->Release();
}

void RenderFrame()
{
    // clear the back buffer to a deep blue
    devcon->ClearRenderTargetView(backbuffer, Colors::DeepPink);

    // do 3D rendering on the back buffer here

    // switch the back buffer and the front buffer
    swapchain->Present(0, 0);
}