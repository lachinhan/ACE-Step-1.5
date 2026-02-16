@echo off
title ACE-Step Server (NO UPDATE)
echo Dang khoi dong ACE-Step (Che do on dinh)...
echo --------------------------------------------

:: Lenh nay chay truc tiep, bo qua moi buoc kiem tra update
uv run acestep --port 8001 --enable-api --backend pt --server-name 127.0.0.1 --offload_dit_to_cpu True

pause