#include "certinfo.h"
// included in cerinfo.h #include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>

// https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html
static size_t wrfu(void *ptr, size_t size, size_t nmemb, void *userdata) {
  (void)ptr;       // Points to the delivered data.
  (void)userdata;  // User-provided data pointer: used to share state between
                   // callbacks and your main program.
  return size * nmemb;
}

CURLcode get_cert_info(const char *url, struct curl_certinfo **result) {
  CURL *curl = curl_easy_init();
  CURLcode res = CURLE_OK;

  if (!curl) return CURLE_FAILED_INIT;

  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, wrfu);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_CERTINFO, 1L);

  struct curl_certinfo *certinfo;
  res = curl_easy_perform(curl);
  // If aok, curl returns CURLE_OK, which is 0, and 0 is "false" in C.
  if (!res && !curl_easy_getinfo(curl, CURLINFO_CERTINFO, &certinfo)) {
    printf("Inside function: %d certs\n", certinfo->num_of_certs);
    *result = malloc(sizeof(struct curl_certinfo));
    (*result)->num_of_certs = certinfo->num_of_certs;
    (*result)->certinfo =
        malloc(sizeof(struct curl_slist *) * certinfo->num_of_certs);

    for (int i = 0; i < certinfo->num_of_certs; i++) {
      struct curl_slist *current = NULL;
      for (struct curl_slist *src = certinfo->certinfo[i]; src;
           src = src->next) {
        struct curl_slist *new_item = curl_slist_append(current, src->data);
        if (!current) current = new_item;
      }
      (*result)->certinfo[i] = current;
    }
  }

  curl_easy_cleanup(curl);
  return res;
}

void free_cert_info(struct curl_certinfo *info) {
  for (int i = 0; i < info->num_of_certs; i++) {
    curl_slist_free_all(info->certinfo[i]);
  }
  free(info->certinfo);
  free(info);
}