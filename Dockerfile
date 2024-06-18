FROM fedora:34 as deploy

# install required software
RUN dnf makecache -y \
    && dnf update -y \
    && dnf install -y ca-certificates sudo which xterm unzip sssd-client acl \
    && dnf clean all \
    && rm -rf /var/cache/yum

# system settings and permissions
COPY src/nofiles.conf /etc/security/limits.d/

## see https://bugzilla.redhat.com/show_bug.cgi?id=1773148
RUN touch /etc/sudo.conf && echo "Set disable_coredump false" > /etc/sudo.conf

RUN dnf install -y python3-notebook
RUN pip install jupyterlab
RUN dnf -y groupinstall "Development Tools"
RUN dnf install -y tcsh
RUN dnf -y install pip
RUN dnf install -y python3-tkinter

RUN pip install emcee numba numpy scipy astropy matplotlib scikit-image
RUN dnf -y install sextractor
RUN pip install vos cadctap cadcdata

COPY src/ds9 /usr/local/bin/

WORKDIR /opt
COPY ./ /opt/trippy
WORKDIR /opt/trippy
RUN rm -rf env
RUN python3 setup.py install

WORKDIR /
RUN mkdir /skaha
COPY src/startup.sh /skaha/startup.sh


# Two build sets, deploy and test
FROM deploy as test
RUN echo "Adding a test user to run local testing"
RUN mkdir -p /arc/home
RUN groupadd -g 1001 testuser
RUN useradd -u 1001 -g 1001 -s /bin/bash -d /arc/home/testuser -m testuser
WORKDIR /arc/home/testuser
COPY trippy/tests ./tests
RUN chown -R 1001 tests
ENTRYPOINT ["/skaha/startup.sh"]
