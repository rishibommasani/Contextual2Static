import regex as re
import sys
pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
text = 'The construction affable roulette disorganization unorganized hyperbolic teacher vehement argumentation insulting running antidisestablishmentarianism.' 

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))

if __name__ == '__main__':
	bpe_tokens = []
	for i, token in enumerate(re.findall(pat, text)):
		print(i)
		if sys.version_info[0] == 2:
			token = ''.join(bytes_to_unicode()[ord(b)] for b in token)
		else:
			token = ''.join(bytes_to_unicode()[b] for b in token.encode('utf-8'))
		bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
	print(len(bpe_tokens))

