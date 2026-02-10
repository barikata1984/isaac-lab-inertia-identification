# =============================================================================
# Isaac Sim + Isaac Lab Development Container
# =============================================================================
# Base: NVIDIA Isaac Sim official container
# Adds:  Isaac Lab (from source), zsh, dev tooling, non-root user
# =============================================================================

ARG ISAAC_SIM_VERSION=5.0.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Base image may set a non-root USER; switch to root for build steps
USER root

# ---- System packages --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Shell & terminal
        zsh \
        tmux \
        # VCS & build essentials
        git \
        git-lfs \
        curl \
        wget \
        unzip \
        build-essential \
        cmake \
        # Editor (fallback)
        nano \
        vim \
        # Python venv support (for system python if ever needed)
        python3-venv \
        # USB / serial device access
        usbutils \
        udev \
        libusb-1.0-0-dev \
        # Misc
        sudo \
        locales \
        ca-certificates \
        openssh-client \
        less \
        jq \
        gosu \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8  LC_ALL=en_US.UTF-8

# ---- Install oh-my-zsh (shared skeleton; user copy happens at entrypoint) ---
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
    && cp -r /root/.oh-my-zsh /etc/skel/.oh-my-zsh

# ---- Isaac Lab installation -------------------------------------------------
ARG ISAAC_LAB_VERSION=v2.3.0
ARG ISAAC_LAB_REPO=https://github.com/isaac-sim/IsaacLab.git

# Workaround: setuptools 82.0.0 (2026-02-08) removed pkg_resources, but
# flatdict 4.0.1 (unmaintained Isaac Lab dependency) imports it in setup.py.
# Constrain setuptools<82 in all pip build-isolation environments.
RUN echo "setuptools<82" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN git clone --branch ${ISAAC_LAB_VERSION} --depth 1 ${ISAAC_LAB_REPO} /opt/isaac-lab \
    && ln -s /isaac-sim /opt/isaac-lab/_isaac_sim \
    && cd /opt/isaac-lab \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -e /opt/isaac-lab/source/isaaclab \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -e /opt/isaac-lab/source/isaaclab_tasks \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -e /opt/isaac-lab/source/isaaclab_rl

# ---- Additional pip packages (from separate requirements file) ---------------
COPY docker/requirements.txt /tmp/requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ---- Non-root user setup (UID/GID injected at build time) -------------------
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=developer

RUN if getent group ${HOST_GID} >/dev/null 2>&1; then \
        existing_grp=$(getent group ${HOST_GID} | cut -d: -f1); \
        [ "$existing_grp" != "${HOST_USER}" ] && groupmod -n ${HOST_USER} "$existing_grp"; \
    else \
        groupadd -g ${HOST_GID} ${HOST_USER}; \
    fi \
    && if getent passwd ${HOST_UID} >/dev/null 2>&1; then \
         existing=$(getent passwd ${HOST_UID} | cut -d: -f1); \
         usermod -l ${HOST_USER} -s /bin/zsh -g ${HOST_GID} -d /home/${HOST_USER} -m "$existing"; \
       else \
         useradd -m -s /bin/zsh -u ${HOST_UID} -g ${HOST_GID} ${HOST_USER}; \
       fi \
    && echo "${HOST_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${HOST_USER} \
    && chmod 0440 /etc/sudoers.d/${HOST_USER} \
    # Give the user access to Isaac Sim & Lab directories
    && chown -R ${HOST_UID}:${HOST_GID} /opt/isaac-lab \
    && chmod -R a+rX /isaac-sim

# ---- Entrypoint script ------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---- Default environment variables ------------------------------------------
ENV ISAAC_SIM_PATH=/isaac-sim
ENV ISAAC_LAB_PATH=/opt/isaac-lab
ENV ISAACSIM_PATH=/isaac-sim

# Default workspace
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["zsh"]
