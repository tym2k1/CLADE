{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.zlib
    pkgs.libGL
  ];

  shellHook = ''
    # put zlib's lib directory first, then the directory containing libstdc++.so
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$(dirname "$(${pkgs.gcc}/bin/gcc -print-file-name=libstdc++.so)"):${pkgs.libGL}/lib:$LD_LIBRARY_PATH"
  '';
}
