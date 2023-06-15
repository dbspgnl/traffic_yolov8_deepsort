@echo off
IF "%1" == "save" (
	IF "%2" == "show" (
		GOTO save_show
	) ELSE (
		GOTO save
	)
) ELSE (
	IF "%2" == "save" (
		GOTO save_show
	) ELSE (
		GOTO show
	)
)

:save_show
python predict.py model=best_m_e2000.pt source="10s.mp4" imgsz=5760 name=measong_t save=true show=true
goto end

:show
python predict.py model=best_m_e2000.pt source="10s.mp4" imgsz=5760 name=measong_t save=false show=true
goto end

:save
python predict.py model=best_m_e2000.pt source="10s.mp4" imgsz=5760 name=measong_t save=true show=false
goto end

:end