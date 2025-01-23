from _certinfo import ffi, lib
import atexit

def get_certificate_info(url):
    cert_info = ffi.new("struct curl_certinfo **")
    result = lib.get_cert_info(url.encode(), cert_info)
    
    if result != 0 or not cert_info[0]:
        return None
        
    try:
        certs = []
        for i in range(cert_info[0].num_of_certs):
            cert_data = []
            slist = cert_info[0].certinfo[i]
            while slist:
                cert_data.append(ffi.string(slist.data).decode('utf-8'))
                slist = slist.next
            certs.append(cert_data)
        return certs
    finally:
        lib.free_cert_info(cert_info[0])

#lib.curl_global_init(0x03)  # CURL_GLOBAL_DEFAULT
atexit.register(lib.curl_global_cleanup)