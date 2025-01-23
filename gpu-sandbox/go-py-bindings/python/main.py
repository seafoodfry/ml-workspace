import certinfo


cert_info = certinfo.get_certificate_info("https://api.seafoodfry.ninja/")

print(f"Found {len(cert_info)} certificates:")
for i, cert in enumerate(cert_info):
    print(f"Certificate {i + 1}:")
    for line in cert:
        print(line)
