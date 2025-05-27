{ pkgs, lib, config, inputs, ... }:

{
  cachix.enable = false;
  
  # https://devenv.sh/packages/
  packages = [ pkgs.figlet pkgs.git pkgs.rustup ];

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
  cargo test
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;
}
