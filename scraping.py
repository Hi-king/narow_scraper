# -*- coding: utf-8 -*-
import json
import os
import re
import urllib.error
import urllib.request
import string

from bs4 import BeautifulSoup

ROOTDIR = os.path.dirname(__file__)
SAVEDIR = os.path.join(ROOTDIR, "dataset")


def save(book_id: str) -> None:
    url = "http://ncode.syosetu.com/novelview/infotop/ncode/{}/".format(book_id)
    try:
        resource = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        return
    soup = BeautifulSoup(resource, "html.parser")
    info_soup = soup.find("table", {"id": "noveltable1"})  # type: BeautifulSoup
    if info_soup is None:  # R18
        return
    info_list = list(info_soup.find_all("tr")) # type: List[BeautifulSoup]
    summary_line_soup = info_list[0]
    genre_line_soup = info_list[-1]
    summary = summary_line_soup.find("td").text
    genre = genre_line_soup.find("td").text
    matcher = re.match('(.*)〔(.*)〕', genre)
    genre_small, genre_large = matcher.groups()

    with open(os.path.join(SAVEDIR, "{}.json".format(book_id)), "w+") as f:
        json.dump({
            "genre_large": genre_large,
            "genre_small": genre_small,
            "summary": summary
        }, f)


for major_code in ["c", "d"]:
    for minor_code in string.ascii_lowercase:
        for i in range(10000):
            print("n{:04}{}{}".format(i, major_code, minor_code))
            save("n{:04}{}{}".format(i, major_code, minor_code))
