"""Step 1: text to token ids and back.

The kit uses the simplest tokenizer that can represent any text: one token per UTF-8 byte, so the
vocabulary has exactly 256 entries and no text is ever "unknown". Real models use larger vocabularies
(byte-pair encoding, BPE) so that one token covers several bytes; see "Going further" in the workbook.
"""

from __future__ import annotations


class ByteTokenizer:
    """One token per UTF-8 byte (vocabulary size 256). Lossless for any text."""

    name = "byte"
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        """Token ids (each 0..255) for ``text``."""
        #lab:begin 1 encode text as UTF-8 byte values
        return list(text.encode("utf-8"))
        #lab:stub
        #> # TODO(step 1): return the UTF-8 bytes of `text` as a list of ints in 0..255.
        #> raise NotImplementedError("TODO(step 1): ByteTokenizer.encode")
        #lab:end

    def decode(self, ids: list[int]) -> str:
        """Text for ``ids``. Invalid UTF-8 (for example half of a multi-byte character, which a model
        can generate) becomes U+FFFD instead of raising."""
        #lab:begin 1 decode byte values back to text
        return bytes(ids).decode("utf-8", errors="replace")
        #lab:stub
        #> # TODO(step 1): turn the ids back into bytes and decode them as UTF-8. A model can produce
        #> # invalid UTF-8, so replace bad bytes instead of raising.
        #> raise NotImplementedError("TODO(step 1): ByteTokenizer.decode")
        #lab:end
