Running the `mila init` command with this initial content:

```
# a comment
Host foo
    HostName foobar.com

```

and these user inputs: ('y', 'bob_mila\r', 'n', 'y')
leads the following ssh config file:

```
# a comment
Host foo
    HostName foobar.com

Host mila
    HostName login.server.mila.quebec
    PreferredAuthentications publickey,keyboard-interactive
    Port 2222
    ServerAliveInterval 120
    ServerAliveCountMax 5
    User bob_mila

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
    User bob_mila

Host *.server.mila.quebec !*login.server.mila.quebec
    HostName %h
    ProxyJump mila
    User bob_mila

Host cn-????
    ProxyJump mila
    User bob_mila
```
