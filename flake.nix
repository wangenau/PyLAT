{
  description = "PyLAT - Python LAMMPS Analysis Tools.";

  inputs = { nixpkgs.url = "nixpkgs/nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };

      pyEnv = pkgs.python3.withPackages (p:
        with p; [
          matplotlib
          numba
          numpy
          pip
          scipy
        ]);

    in
    {
      devShells."${system}".default = with pkgs;
        mkShell {
          buildInputs = [
            pyEnv
            ruff
          ];

          shellHook = ''
            pip install -e . --prefix "$TMPDIR"
            export PYTHONPATH="$(pwd):$PYTHONPATH"
            export MPLBACKEND="TKAgg"
          '';
        };
    };
}
