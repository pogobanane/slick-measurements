{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, ...} @ inputs: let
    pkgs = inputs.nixpkgs.legacyPackages.x86_64-linux;
  in {

    packages.x86_64-linux.hello = pkgs.hello;

    packages.x86_64-linux.default = self.packages.x86_64-linux.hello;

    devShells.x86_64-linux.default = pkgs.mkShell {
      name = "hello";
      buildInputs = with pkgs.python3.pkgs; [
        black # auto formatting
        flake8 # annoying "good practice" annotations
        mypy # static typing
        pkgs.ruff # language server ("linting")

        numpy
        matplotlib
        seaborn
        tqdm
        scipy

        bpython
        ptpython
        ipykernel
      ];
    };
  };
}
