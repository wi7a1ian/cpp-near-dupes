# cpp-near-dupes
Implementation of near duplicate algorithm for documents

- Clone with submodules: `git clone ... --recurse-submodules` or run `git submodule update --init --recursive`.
- Build using cmake
- Run `rm /tmp/ndd-enron-100k/* && ./build/src/NearDupes.App`
  > min-hash time in seconds : 46 sec
  > lsh time in seconds : 132 sec