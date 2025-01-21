#include <curl/curl.h>
#include <stdio.h>

// https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html
static size_t wrfu(void *ptr, size_t size, size_t nmemb, void *userdata) {
  (void)ptr;       // Points to the delivered data.
  (void)userdata;  // User-provided data pointer: used to share state between
                   // callbacks and your main program.
  return size * nmemb;
}

int main(void) {
  CURL *curl;
  CURLcode res;

  curl_global_init(CURL_GLOBAL_DEFAULT);

  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.seafoodfry.ninja/");

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, wrfu);

    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_CERTINFO, 1L);

    res = curl_easy_perform(curl);
    if (!res) {
      struct curl_certinfo *certinfo;

      res = curl_easy_getinfo(curl, CURLINFO_CERTINFO, &certinfo);
      if (!res && certinfo) {
        int i;

        printf("%d certs!\n", certinfo->num_of_certs);

        for (i = 0; i < certinfo->num_of_certs; i++) {
          // See
          // https://github.com/curl/curl/blob/7c039292ad955db327dce4e9e6a3f612ddef3c58/include/curl/curl.h#L2762-L2766
          struct curl_slist *slist;

          for (slist = certinfo->certinfo[i]; slist; slist = slist->next)
            printf("%s\n", slist->data);
        }
      }
    }

    curl_easy_cleanup(curl);
  }

  curl_global_cleanup();

  return 0;
}