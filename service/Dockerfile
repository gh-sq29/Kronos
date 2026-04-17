FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

WORKDIR /app

# PyTorch is already in the base image — only install lightweight web/db deps.
# Source files are bind-mounted from the host at runtime (git pull → restart = deploy).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# Parent repo (model/) is bind-mounted at /kronos at runtime
ENV PYTHONPATH=/kronos

EXPOSE 16300

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
