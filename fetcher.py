import requests
import re

from lxml import etree


class Fetcher:

    def __init__(self, root, artist):
        self.root = root
        self.artist = artist
        self.songs = set()
        self.lyrics = []

    def get_songs(self, albums):

        for album in albums:
            print(album)
            path_album = root + "albums/" + artist + "/" + album
            page = requests.get(path_album).content.decode("utf-8")
            html = etree.HTML(page)
            items = html.xpath("//div[@class='chart_row-content']")
            for item in items:
                child = item.getchildren()[0]
                if "href" in child.attrib:
                    song_link = child.attrib['href']
                    if re.search(artist, song_link):
                        self.songs.add(song_link)

    def get_lyrics(self):

        for song in self.songs:
            lyric = ""
            print(song)
            page = requests.get(song).content.decode("utf-8")
            html = etree.HTML(page)
            item = html.xpath("//div[@class='lyrics']")[0]
            children = []
            for child in item.getchildren():
                if child.tag == "p":
                    children.append(child)

            for child in children:
                for grandchild in child.getchildren():
                    if grandchild.text is not None:
                        lyric += re.sub("\n", " *BREAK* ", grandchild.text)
                    elif grandchild.tail is not None:
                        lyric += re.sub("\n", " *BREAK* ", grandchild.tail)

                    if grandchild.tag == "a":
                        for descendant in grandchild.getchildren():
                            if descendant.text is not None:
                                lyric += re.sub("\n", " *BREAK* ", descendant.text)
                            elif descendant.tail is not None:
                                lyric += re.sub("\n", " *BREAK* ", descendant.tail)


            self.lyrics.append(lyric)

    def print_lyrics(self, path):

        with open(path, "w+", encoding="utf-8") as writer:
            for lyric in self.lyrics:
                writer.write(lyric + "\n")

    def fetch(self, albums, path):

        self.get_songs(albums)
        self.get_lyrics()
        self.print_lyrics(path)

if __name__ == "__main__":

    root = 'https://genius.com/'
    artist = 'Iam'
    fetcher = Fetcher(root, artist)
    albums = ["De-la-planete-mars",
              "Ombre-est-lumiere",
              "L-ecole-du-micro-d-argent",
              "Revoir-un-printemps",
              "Saison-5",
              "Arts-martiens",
              "Revolution",
              "Iam"]
    path = "/Users/jerome/Documents/garnieje/deep-french-rap/data/lyrics_" + artist + ".csv"
    fetcher.fetch(albums, path)

