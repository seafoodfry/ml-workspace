#pragma once

#include <curl/curl.h>

// Gets certificate information for the given URL.
CURLcode get_cert_info(const char *url, struct curl_certinfo **result);

// Frees memory allocated for certificate information.
void free_cert_info(struct curl_certinfo *info);