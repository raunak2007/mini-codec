import modal, subprocess
app = modal.App("codec-stage")
vol = modal.Volume.from_name("codec-data", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("curl")

@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def stage(split: str):
    url = f"https://www.openslr.org/resources/141/{split}.tar.gz"
    subprocess.run(f"cd /data && curl -sSL -O {url} && tar -xzf {split}.tar.gz && rm {split}.tar.gz",
                   shell=True, check=True)
    vol.commit()
