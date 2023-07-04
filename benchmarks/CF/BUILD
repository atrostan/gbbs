licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CF",
    hdrs = ["CF.h"],
    deps = [
        "//gbbs",
        "//gbbs:edge_map_reduce",
        "//gbbs/helpers:sparse_table",
    ],
)

cc_binary(
    name = "CF_main",
    srcs = ["CF.cc"],
    deps = [":CF"],
)
