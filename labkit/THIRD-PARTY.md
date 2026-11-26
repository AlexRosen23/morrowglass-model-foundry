# Third-party software and text

## Software

The kit's code imports these open-source Python packages. They are **not included** in the kit: you
install them with `pip` from the Python Package Index (or PyTorch's own index). Each is licensed to
you by its authors under its own licence.

| Package | Tested version | Licence | Used for |
|---|---|---|---|
| PyTorch (`torch`) | 2.15.0 | BSD-3-Clause | tensors, autograd, layers, AdamW |
| pytest | 9.2.0 | MIT | running the tests |

PyTorch installs its own dependencies (for example `numpy` is optional and not used by the kit);
`pip show torch` lists them. The kit's own code only imports and calls these packages; it does not
copy or modify their source. One test compares your attention with PyTorch's built-in
`scaled_dot_product_attention` as a reference.

## Practice text

`data/shakespeare-train.txt` (240 KB) and `data/shakespeare-val.txt` (40 KB) are excerpts of the
works of William Shakespeare, which are in the public domain. They come from a free public-domain
e-text of the Complete Works. The distributor's header, footer, licence text and every reference to
the distributor were removed, which its terms require when the text is passed on without its
trademark; typographic quotes and dashes were changed to plain ASCII, line endings and trailing spaces were
normalised, and each file is cut at a line break. The words themselves are unchanged. The
validation file comes from different works than the training file, so the model is scored on plays
it never saw.
