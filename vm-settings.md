# VM Settings
This document guide you to set-up a linux virtual machine on your Mac.

1. Install `multipass`
    ```bash
    brew install multipass
    ```
2. Then create an Ubuntu@22.04 instance named `primary`.
    ```bash
    multipass launch 22.04 -n primary -c 8 -m 32G -d 60G
    multipass exec primary sudo apt update
    multipass exec primary sudo apt upgrade
    multipass restart primary
    multipass shell primary
    ```
    You can change the name if you want :)
3. (optional) If you want to mount a folder in VM to your preferred directory in your host machine,
    ```bash
    multipass mount <host-machine-path> primary:<vm-path>
    ```

# SSH Setting
If you want to use a remote VS code on your host machine, set up `ssh` as follows.

1. Copy your public key to VM. You can find your public key by
    ```bash
    cat ~/.ssh/id_rsa.pub
    ```
    The content should be put at `primary:/home/<username>/.ssh/authorized_keys`.

2. Get VM's local ip-address. Type from your host machine.
    ```bash
    $ multipass list
    > Name                    State             IPv4             Image
      primary                 Running           192.168.64.26    Ubuntu 22.04 LTS
    ```

Now, follow `README.md` to install everything.

