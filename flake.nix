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

          # Dependencies not available in nixpkgs — pre-fetch for FetchContent
          qdmi-src = pkgs.fetchFromGitHub {
            owner = "Munich-Quantum-Software-Stack";
            repo = "qdmi";
            rev = "0f7e08c58b72800d1022a01cfb618af67b9a9c30"; # v1.3.0
            hash = "sha256-DrL5HAo/ZvPuKID2R5pV9v8L/ZjKUDzwBKL6OdRIK84=";
          };

          capnproto-src = pkgs.fetchFromGitHub {
            owner = "capnproto";
            repo = "capnproto";
            rev = "v1.3.0";
            hash = "sha256-fvZzNDBZr73U+xbj1LhVj1qWZyNmblKluh7lhacV+6I=";
          };

          jeff-src = pkgs.fetchFromGitHub {
            owner = "unitaryfoundation";
            repo = "jeff";
            rev = "jeff-v0.2.0";
            hash = "sha256-WQ45S2Dm3PjL1q6oJ05kRdPwHWMxR2fgTetydDEtmP8=";
          };

          jeff-mlir-src = pkgs.applyPatches {
            name = "jeff-mlir-patched";
            src = pkgs.fetchFromGitHub {
              owner = "PennyLaneAI";
              repo = "jeff-mlir";
              rev = "v0.2.0";
              hash = "sha256-Ajj3iMNIxcZ2Yz21oXK/s3OaoxsZDnEm4ha/258KDoU=";
            };
            postPatch = ''
              cp ${./cmake/patches/jeff-mlir-SetupMLIR.cmake} cmake/SetupMLIR.cmake
            '';
          };
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

          # Run the hooks with `nix fmt`. Skip clang-tidy here (slow, not a
          # formatter); it still runs under `nix flake check`.
          formatter = pkgs.writeShellScriptBin "pre-commit-run" ''
            export SKIP=clang-tidy
            ${pkgs.lib.getExe config.pre-commit.settings.package} \
              run --all-files --config ${config.pre-commit.settings.configFile}
          '';

          packages.mqt = pkgs.llvmPackages_22.stdenv.mkDerivation {
            src = pkgs.lib.cleanSource ./.;
            pname = "mqt";
            version = "3.6.1";
            nativeBuildInputs = with pkgs; [
              pkg-config
              cmake
              ninja
              ccache
              patched-mlir
              patched-mlir.dev
              llvmPackages_22.libllvm.dev
              llvmPackages_22.tblgen
            ];
            buildInputs = with pkgs; [
              nlohmann_json
              spdlog
              gtest
              eigen_5
              boost
              # capnproto — built from source via FetchContent (version must match jeff)
            ];
            cmakeFlags = [
              "-DBUILD_MQT_CORE_MLIR=ON"
              "-DBUILD_MQT_CORE_TESTS=ON"
              "-DUSE_SYSTEM_BOOST=ON"
              "-DBUILD_JEFF_MLIR_TRANSLATION=ON"
              "-DFETCHCONTENT_SOURCE_DIR_JEFF=${jeff-src}"
              "-DFETCHCONTENT_SOURCE_DIR_CAPNPROTO=${capnproto-src}"
              "-DFETCHCONTENT_SOURCE_DIR_QDMI=${qdmi-src}"
              "-DFETCHCONTENT_SOURCE_DIR_JEFF-MLIR=${jeff-mlir-src}"
              "-DFETCHCONTENT_SOURCE_DIR_EIGEN=${pkgs.eigen_5.src}"
              "-DMLIR_TABLEGEN_EXE=${pkgs.llvmPackages_22.tblgen}/bin/mlir-tblgen"
              # Skip libstdc++ version checks from HandleLLVMOptions.cmake.
              # The Nix clang wrapper uses GCC 15's libstdc++ but CMake's
              # check_cxx_source_compiles may not pick up the wrapper flags.
              # These checks are redundant for consumers of pre-built MLIR.
              "-DLLVM_LIBSTDCXX_MIN=ON"
              "-DLLVM_LIBSTDCXX_SOFT_ERROR=ON"
              "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF"
              # Disable LTO — nixpkgs MLIR static libs aren't built with LTO,
              # causing linker errors when mixing LTO and non-LTO object files.
              "-DENABLE_IPO=OFF"
              # MLIR static libs have circular deps; linker needs multiple passes.
              "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--start-group"
              # No .git in Nix store; set version explicitly
              "-DMQT_CORE_VERSION=3.6.1"
            ];
            # add_mlir_tool marks mqt-cc EXCLUDE_FROM_ALL — build it explicitly
            buildPhase = ''
              runHook preBuild
              cmake --build . --target mqt-cc --parallel $NIX_BUILD_CORES
              runHook postBuild
            '';

            # Build all targets (including tests) then run ctest
            doCheck = true;
            checkPhase = ''
              cmake --build . --parallel $NIX_BUILD_CORES
              ctest -C Release -j $NIX_BUILD_CORES --output-on-failure
            '';

            # mqt-cc has no cmake install rules; install manually
            installPhase = ''
              mkdir -p $out/bin
              cp mlir/tools/mqt-cc/mqt-cc $out/bin/
            '';
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
