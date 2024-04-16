# cpp-near-dupes
Implementation of near duplicate algorithm for documents

- Clone with submodules: `git clone ... --recurse-submodules` or run `git submodule update --init --recursive`.
- Build using cmake
- Run `rm /tmp/ndd-cache/* && ./build/src/NearDupes.App`
  ```
  min-hash took 8 sec
  lsh took 10 sec
  ```