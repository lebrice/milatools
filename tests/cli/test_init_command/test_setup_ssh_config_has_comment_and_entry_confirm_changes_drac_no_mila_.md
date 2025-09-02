Running the `mila init` command with this initial content:

```
# a comment
Host foo
  HostName foobar.com

# another comment

```

and these user inputs: ('n', 'y', 'bob_drac\r', 'y')
leads the following ssh config file:

```
# a comment
Host foo
  HostName foobar.com

# another comment

Host beluga cedar graham narval niagara rorqual fir nibi tamia killarney vulcan
  HostName %h.alliancecan.ca
  ControlMaster auto
  ControlPath ~/.cache/ssh/%r@%h:%p
  ControlPersist yes
  User bob_drac

Host !beluga  bc????? bg????? bl?????
  ProxyJump beluga
  User bob_drac

Host !cedar   cdr? cdr?? cdr??? cdr????
  ProxyJump cedar
  User bob_drac

Host !graham  gra??? gra????
  ProxyJump graham
  User bob_drac

Host !narval  nc????? ng?????
  ProxyJump narval
  User bob_drac

Host !niagara nia????
  ProxyJump niagara
  User bob_drac

Host rc????? rg????? rl?????
  ProxyJump rorqual
  User bob_drac

Host fc????? fb?????
  ProxyJump fir
  User bob_drac

Host c? c?? c??? g? g?? l? l?? m? m?? u?
  ProxyJump nibi
  User bob_drac

Host tg????? tc?????
  ProxyJump tamia
  User bob_drac

Host kn???
  ProxyJump killarney
  User bob_drac

Host rack??-??
  ProxyJump vulcan
  User bob_drac
```
