#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/21
from __future__ import unicode_literals
from __future__ import division


def len_diff(s1, s2):
    return abs(len(s1) - len(s2))


def len_diff_ratio(s1, s2):
    return 2 * abs(len(s1) - len(s2)) / (len(s1) + len(s2))


def shingle_similarity(s1, s2, size=1):
    """Shingle similarity of two sentences."""
    def get_shingles(text, size):
        shingles = set()
        for i in range(0, len(text) - size + 1):
            shingles.add(text[i:i + size])
        return shingles

    def jaccard(set1, set2):
        x = len(set1.intersection(set2))
        y = len(set1.union(set2))
        return x, y

    x, y = jaccard(get_shingles(s1, size), get_shingles(s2, size))
    return x / float(y) if (y > 0 and x > 2) else 0.0


def common_words(s1, s2):
    s1_common_cnt = len([w for w in s1 if w in s2])
    s2_common_cnt = len([w for w in s2 if w in s1])
    return (s1_common_cnt + s2_common_cnt) / (len(s1) + len(s2))


def tf_idf():
    pass


def wmd():
    pass


if __name__ == '__main__':
    s1 = '怎么更改花呗手机号码'
    s2 = '我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号'
    print(len_diff(s1, s2))
    print(shingle_similarity(s1, s2))
    print(shingle_similarity(s1, s2, 2))
    print(shingle_similarity(s1, s2, 3))

