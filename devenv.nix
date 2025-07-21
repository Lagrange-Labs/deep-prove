{ pkgs, lib, config, inputs, ... }:

{
  cachix.enable = false;

  packages = [
    # General dev.
    pkgs.git pkgs.openssl
    # Rust
    pkgs.rustup pkgs.protobuf pkgs.openssl
  ];

  env = {
    OPENSSL_DEV = pkgs.openssl.dev;
  };

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
  cargo test
  '';

  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ''
    matplotlib
    numpy
    onnx
    psutil
    pandas
    scikit-learn
    tabulate
    torch
    torchvision
    tqdm
    '';
  };

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;
  git-hooks.hooks = {
    # actionlint.enable = true;
    check-merge-conflicts.enable = true;
    ripsecrets.enable = true;
    rustfmt = {
      enable = false;
      settings.color = "auto";
    };
    black = {
      enable = true;
    };
    taplo = {
      enable = true;
    };
    typos = {
      enable = true;
      settings = {
        format = "brief";
        write = true;
        configPath = "typos.toml";
      };
    };
  };
}
