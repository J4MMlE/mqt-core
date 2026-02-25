{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    git-hooks.url = "github:cachix/git-hooks.nix";
    mini-compile-commands = {
      url = "github:danielbarter/mini_compile_commands";
      flake = false;
    };
  };

  outputs =
    inputs@{
      flake-parts,
      systems,
      mini-compile-commands,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      # inherit systems;
      systems = [ "x86_64-linux" ];
      imports = [
        inputs.git-hooks.flakeModule
        # To import a flake module
        # 1. Add foo to inputs
        # 2. Add foo as a parameter to the outputs function
        # 3. Add here: foo.flakeModule

      ];
      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          ...
        }:
        let
          patched-mlir = pkgs.llvmPackages_22.mlir.dev.overrideAttrs (oldAttrs: {
            # only use 6 cores on my build machine
            ninjaFlags = [ "-j6" ];

            # mlir is build standalone from the llvm-project repository.
            # Somehow when building mlir standalone this leads to an unset PACKAGE_VERSION for mlir.
            # setting PACKAGE_VERSION explicitly to the package version is a workaround.
            cmakeFlags = oldAttrs.cmakeFlags ++ [
              (pkgs.lib.cmakeFeature "PACKAGE_VERSION" oldAttrs.version)
            ];
          });
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfreePredicate =
              pkg:
              builtins.elem (inputs.nixpkgs.lib.getName pkg) [
                "claude-code"
              ];
          };

          # Per-system attributes can be defined here. The self' and inputs'
          # module parameters provide easy access to attributes of the same
          # system.
          pre-commit = {
            settings.hooks.clang-tidy.enable = true;
            settings.hooks.clang-format.enable = true;
            check.enable = true;
          };

          packages.mqt = pkgs.stdenv.mkDerivation {
            src = ./.;
            pname = "mqt";
            version = "1.2.3";
            nativeBuildInputs = with pkgs; [
              pkg-config
              cmake
              # llvmPackages_21.llvm.dev
              # llvmPackages_21.llvm
              # llvmPackages_21.tblgen
              patched-mlir
              patched-mlir.dev
              triton-llvm
              # llvmPackages_21.mlir.dev
              # llvmPackages_21.mlir
            ];
            # cmakeFlags = [
            #   "-DMLIR_DIR=${pkgs.llvmPackages_21.mlir.dev}/lib/cmake/mlir"
            #   "-DLLVM_DIR=${pkgs.llvmPackages_21.llvm.dev}/lib/cmake/llvm"
            #   "-DMLIR_VERSION=${pkgs.llvmPackages_21.mlir.version}"
            # ];
          };
          # Equivalent to  inputs'.nixpkgs.legacyPackages.hello;
          packages.default = self'.packages.mqt;

          devShells.default =
            let
              mcc-env = (pkgs.callPackage mini-compile-commands { }).wrap pkgs.llvmPackages_22.stdenv;
            in
            pkgs.mkShell.override { stdenv = mcc-env; } {
              # pkgs.mkShell.override { inherit (pkgs.llvmPackages_21) stdenv; } {
              inputsFrom = [ self'.packages.mqt ];
              nativeBuildInputs = with pkgs; [
                pkg-config
                cmake
                llvmPackages_22.libllvm.dev # needed for llvm
                # llvmPackages_21.llvm.dev # needed for llvm
                llvmPackages_22.tblgen # for mlir-tblgen command
                # llvmPackages_21.clang
                llvmPackages_22.clang-tools
                # llvmPackages_21.libcxx
                # llvmPackages_21.libcxxStdenv
                patched-mlir.dev
                lit
                ninja
              ];
              # nativeBuildInputs = with pkgs.llvmPackages_21; [ mlir.dev ];
              #
              #
              packages = with pkgs; [
                nodejs
                bun
                claude-code
                bubblewrap
                # ocaml
                # dune_3
                # ocamlPackages.utop
                # ocamlPackages.lsp
                # ocamlPackages.ocamlformat
                # ocamlPackages.ocp-indent
                # ocamlPackages.findlib
                # ocamlPackages.merlin
                # ocamlPackages.odoc
                (python3.withPackages (
                  python-pkgs: with python-pkgs; [
                    sympy
                    numpy
                  ]
                ))
                gcovr
                libgccjit # for gcov
                gtest
                socat
              ];

              shellHook = ''
                # export LLVM_EXTERNAL_LIT="${pkgs.triton-llvm}/bin/llvm-lit"
                # export MLIR_TABLEGEN_EXE="${pkgs.llvmPackages_22.tblgen}/bin/mlir-tblgen"
                export MLIR_DIR="${patched-mlir.dev}/lib/cmake/mlir"
                # export LLVM_DIR="${pkgs.llvmPackages_22.llvm.dev}/lib/cmake/llvm"
                #   export MLIR_VERSION="${pkgs.llvmPackages_22.mlir.version}"
              '';
            };
        };
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.

      };
    };
}
