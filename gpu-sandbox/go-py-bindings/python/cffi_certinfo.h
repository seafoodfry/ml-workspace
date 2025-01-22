struct curl_slist {
    char *data;
    struct curl_slist *next;
};

struct curl_certinfo {
    int num_of_certs;
    struct curl_slist **certinfo;
};

typedef int CURLcode;
CURLcode get_cert_info(const char *url, struct curl_certinfo **result);
void free_cert_info(struct curl_certinfo *info);
void curl_global_init(long flags);
void curl_global_cleanup(void);