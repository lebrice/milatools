Running the `mila init` command with this initial content:

```
Host *.server.mila.quebec !*login.server.mila.quebec
  HostName foooobar.com

```

and these user inputs: ['y', 'bob\r', 'y', 'bob\r', 'y']
leads to the following ssh config file:

```
Host *.server.mila.quebec !*login.server.mila.quebec
  HostName %h
  ProxyJump mila
  User bob

Host mila
  HostName login.server.mila.quebec
  PreferredAuthentications publickey,keyboard-interactive
  Port 2222
  ServerAliveInterval 120
  ServerAliveCountMax 5
  User bob

Host mila-cpu
  Port 2222
  ForwardAgent yes
  StrictHostKeyChecking no
  LogLevel ERROR
  UserKnownHostsFile /dev/null
  RequestTTY force
  ConnectTimeout 600
  ServerAliveInterval 120
  ProxyCommand ssh mila "/cvmfs/config.mila.quebec/scripts/milatools/slurm-proxy.sh mila-cpu --mem=8G"
  RemoteCommand /cvmfs/config.mila.quebec/scripts/milatools/entrypoint.sh mila-cpu
  User bob

Host cn-????
  ProxyJump mila
  User bob

Host beluga cedar graham narval niagara rorqual fir nibi tamia killarney vulcan
  HostName %h.alliancecan.ca
  ControlMaster auto
  ControlPath ~/.cache/ssh/%r@%h:%p
  ControlPersist yes
  User bob

Host !beluga  bc????? bg????? bl?????
  ProxyJump beluga
  User bob

Host !cedar   cdr? cdr?? cdr??? cdr????
  ProxyJump cedar
  User bob

Host !graham  gra??? gra????
  ProxyJump graham
  User bob

Host !narval  nc????? ng?????
  ProxyJump narval
  User bob

Host !niagara nia????
  ProxyJump niagara
  User bob

Host rc????? rg????? rl?????
  ProxyJump rorqual
  User bob

Host fc????? fb?????
  ProxyJump fir
  User bob

Host c? c?? c??? g? g?? l? l?? m? m?? u?
  ProxyJump nibi
  User bob

Host tg????? tc?????
  ProxyJump tamia
  User bob

Host kn???
  ProxyJump killarney
  User bob

Host rack??-??
  ProxyJump vulcan
  User bob
```