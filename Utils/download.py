# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import requests, tarfile, io, os, zipfile

from io import BytesIO, SEEK_SET, SEEK_END


class UrlStream:
    def __init__(self, url):
        self._url = url
        headers = requests.head(url).headers
        headers = {k.lower(): v for k,v in headers.items()}
        self._seek_supported = headers.get('accept-ranges')=='bytes' and 'content-length' in headers
        if self._seek_supported:
            self._size = int(headers['content-length'])
        self._curr_pos = 0
        self._buf_start_pos = 0
        self._iter = None
        self._buffer = None
        self._buf_size = 0
        self._loaded_all = False

    def seekable(self):
        return self._seek_supported
    
    def _load_all(self):
        if self._loaded_all:
            return
        self._make_request()
        old_buf_pos = self._buffer.tell()
        self._buffer.seek(0, SEEK_END)
        for chunk in self._iter:
            self._buffer.write(chunk)
        self._buf_size = self._buffer.tell()
        self._buffer.seek(old_buf_pos, SEEK_SET)
        self._loaded_all = True

    def seek(self, position, whence=SEEK_SET):
        if whence == SEEK_END:
            assert position<=0
            if self._seek_supported:
                self.seek(self._size + position)
            else:
                self._load_all()
                self._buffer.seek(position, SEEK_END)
                self._curr_pos = self._buffer.tell()
        elif whence==SEEK_SET:
            if self._curr_pos != position:
                self._curr_pos = position
                if self._seek_supported:
                    self._iter = None
                    self._buffer = None
                else:
                    self._load_until(position)
                    self._buffer.seek(position)
                    self._curr_pos = position
        else:
            assert "Invalid whence %s" % whence

        return self.tell()

    def tell(self):
        return self._curr_pos

    def _load_until(self, goal_position):
        self._make_request()
        old_buf_pos = self._buffer.tell()
        current_position = self._buffer.seek(0, SEEK_END)

        goal_position = goal_position - self._buf_start_pos
        while current_position < goal_position:
            try:
                d = next(self._iter)
                self._buffer.write(d)
                current_position += len(d)
            except StopIteration:
                break
        self._buf_size = current_position
        self._buffer.seek(old_buf_pos, SEEK_SET)

    def _new_buffer(self):
        remaining = self._buffer.read() if self._buffer is not None else None
        self._buffer = BytesIO()
        if remaining is not None:
            self._buffer.write(remaining)
        self._buf_start_pos = self._curr_pos
        self._buf_size = 0 if remaining is None else len(remaining)
        self._buffer.seek(0, SEEK_SET)
        self._loaded_all = False

    def _make_request(self):
        if self._iter is None:
            h = {
                "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36",
            }
            if self._seek_supported:
                h["Range"] = "bytes=%d-%d" % (self._curr_pos, self._size - 1)

            r = requests.get(self._url, headers=h, stream=True)

            self._iter = r.iter_content(1024 * 1024)
            self._new_buffer()
        elif self._seek_supported and self._buf_size > 128 * 1024 * 1024:
            self._new_buffer()

    def size(self):
        if self._seek_supported:
            return self._size
        else:
            self._load_all()
            return self._buf_size

    def read(self, size=None):
        if size is None:
            size = self.size()

        self._load_until(self._curr_pos + size)
        if self._seek_supported:
            self._curr_pos = min(self._curr_pos+size, self._size)

        return self._buffer.read(size)

    def iter_content(self, block_size):
        while True:
            d = self.read(block_size)
            if not len(d):
                break
            yield d


def download(url, dest=None, extract=True, ignore_if_exists=False):
    """
    Download a file from the internet.

    Args:
        url: the url to download
        dest: destination file if extract=False, or destionation dir if extract=True. If None, it will be the last part of URL.
        extract: extract a tar.gz or zip file?
        ignore_if_exists: don't do anything if file exists

    Returns:
        the destination filename.
    """
    
    base_url = url.split("?")[0]

    if dest is None:
        dest = [f for f in base_url.split("/") if f][-1]

    if os.path.exists(dest) and ignore_if_exists:
        return dest

    stream = UrlStream(url)
    extension = base_url.split(".")[-1].lower()

    if extract and extension in ['gz', 'bz2', 'zip']:
        os.makedirs(dest, exist_ok=True)

        if extension in ['gz', 'bz2']:
            decompressed_file = tarfile.open(fileobj=stream, mode='r|'+extension)
        elif extension=='zip':
            decompressed_file = zipfile.ZipFile(stream, mode='r')
        else:
            assert False, "Invalid extension: %s" % extension

        decompressed_file.extractall(dest)
    else:
        try:
            with open(dest, 'wb') as f:
                for d in stream.iter_content(1024*1024):
                    f.write(d)
        except:
            os.remove(dest)
            raise
    return dest
