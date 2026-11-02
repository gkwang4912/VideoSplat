from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import config


def natural_key(text: str):
    """讓 1, 2, 10 按數字順序排序，而不是 1, 10, 2。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def quote_cmd(args: list[str]) -> str:
    """只用於顯示命令，不影響實際 subprocess 執行。"""
    return subprocess.list2cmdline(args)


def get_brush_help(brush_exe: Path) -> str:
    """
    取得 Brush CLI help，只拿來辨識 CLI 版本與必要參數。
    真正訓練不設定 timeout。
    """
    try:
        result = subprocess.run(
            [str(brush_exe), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        return result.stdout or ""
    except Exception as exc:
        print(f"[警告] 無法讀取 Brush --help：{exc}")
        return ""


def detect_steps_flag(help_text: str) -> str:
    """自動兼容不同 Brush 版本的 steps 參數名稱。"""
    if not config.AUTO_DETECT_STEPS_FLAG:
        return config.MANUAL_STEPS_FLAG

    if "--total-train-iters" in help_text:
        return "--total-train-iters"

    if "--total-steps" in help_text:
        return "--total-steps"

    print(
        "[警告] 無法從 Brush --help 判斷 steps 參數，"
        f"改用 MANUAL_STEPS_FLAG={config.MANUAL_STEPS_FLAG}"
    )
    return config.MANUAL_STEPS_FLAG


def validate_brush_flags(help_text: str) -> tuple[bool, str]:
    """確認執行檔確實是支援目前新流程的 Brush CLI。"""
    if not config.REQUIRE_LATEST_BRUSH_FLAGS:
        return True, ""

    if not help_text.strip():
        return False, "讀不到 Brush --help，無法確認 CLI 版本"

    missing = [flag for flag in config.REQUIRED_FLAGS if flag not in help_text]
    if missing:
        return False, (
            "目前 Brush 執行檔缺少新流程需要的參數："
            + ", ".join(missing)
            + "。請確認 BRUSH_EXE 指向新版 brush-cli.exe。"
        )

    return True, ""


def add_optional_arg(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def png_has_alpha(path: Path) -> bool | None:
    """
    快速讀 PNG IHDR 的 color type：4/6 代表含 alpha。
    回傳 None 代表不是可辨識的 PNG header。
    """
    try:
        with path.open("rb") as f:
            header = f.read(26)
        if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        color_type = header[25]
        return color_type in (4, 6)
    except OSError:
        return None


def validate_dataset(dataset_dir: Path) -> tuple[bool, str]:
    if not dataset_dir.is_dir():
        return False, f"找不到資料集資料夾：{dataset_dir}"

    images_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse"

    if not images_dir.is_dir():
        return False, f"缺少 images：{images_dir}"

    if not sparse_dir.is_dir():
        return False, f"缺少 sparse：{sparse_dir}"

    image_files = sorted(
        [p for p in images_dir.rglob("*") if p.is_file()],
        key=lambda p: natural_key(p.name),
    )
    if not image_files:
        return False, f"images 裡沒有任何影像檔：{images_dir}"

    if config.CHECK_PNG_ALPHA:
        pngs = [p for p in image_files if p.suffix.lower() == ".png"]
        if pngs:
            # 抽查前、中、後幾張，避免掃完整資料集造成不必要 I/O。
            sample_indices = sorted(set([0, len(pngs) // 2, len(pngs) - 1]))
            checked = [png_has_alpha(pngs[i]) for i in sample_indices]
            if all(v is False for v in checked):
                return False, (
                    "抽查到的 PNG 都沒有 alpha channel；目前流程要求 RGBA 去背 PNG。"
                    f" 資料夾：{images_dir}"
                )
            if any(v is False for v in checked) and any(v is True for v in checked):
                print("[警告] images 中抽查到 RGB / RGBA PNG 混用，請確認資料一致性。")

    return True, ""


def build_command(
    brush_exe: Path,
    dataset_dir: Path,
    group_name: str,
    output_dir: Path,
    steps_flag: str,
) -> list[str]:
    cmd = [
        str(brush_exe),
        str(dataset_dir),
    ]

    if config.WITH_VIEWER:
        cmd.append("--with-viewer")

    # --------------------------------------------------------
    # 高品質訓練
    # --------------------------------------------------------
    cmd.extend([steps_flag, str(config.TOTAL_STEPS)])
    cmd.extend(["--max-splats", str(config.MAX_SPLATS)])
    cmd.extend(["--max-resolution", str(config.MAX_RESOLUTION)])
    cmd.extend(["--sh-degree", str(config.SH_DEGREE)])
    cmd.extend(["--seed", str(config.SEED)])

    # --------------------------------------------------------
    # RGBA / 去背：這是目前已驗證成功的核心設定
    # --------------------------------------------------------
    cmd.extend(["--alpha-mode", str(config.ALPHA_MODE)])
    cmd.extend(["--match-alpha-weight", str(config.MATCH_ALPHA_WEIGHT)])

    # --------------------------------------------------------
    # 防止 Gaussian 極端扁平，避免後續 ComfyUI SplatToMesh covariance 爆掉
    # --------------------------------------------------------
    cmd.extend(["--min-scale-factor", str(config.MIN_SCALE_FACTOR)])

    # --------------------------------------------------------
    # 延長 densification / growth，提升人物細節
    # --------------------------------------------------------
    cmd.extend(["--growth-stop-iter", str(config.GROWTH_STOP_ITER)])
    add_optional_arg(cmd, "--growth-start-iter", config.GROWTH_START_ITER)
    add_optional_arg(cmd, "--growth-grad-threshold", config.GROWTH_GRAD_THRESHOLD)
    add_optional_arg(cmd, "--growth-select-fraction", config.GROWTH_SELECT_FRACTION)
    add_optional_arg(cmd, "--refine-every", config.REFINE_EVERY)
    add_optional_arg(cmd, "--split-at-screen-size", config.SPLIT_AT_SCREEN_SIZE)

    # 背景設定：預設 None，沿用 Brush 目前預設/成功基準。
    if config.BACKGROUND_COLOR is not None:
        if isinstance(config.BACKGROUND_COLOR, (tuple, list)):
            bg = ",".join(str(x) for x in config.BACKGROUND_COLOR)
        else:
            bg = str(config.BACKGROUND_COLOR)
        cmd.extend(["--background-color", bg])
    add_optional_arg(
        cmd,
        "--background-noise-strength",
        config.BACKGROUND_NOISE_STRENGTH,
    )

    # Dataset
    add_optional_arg(cmd, "--max-frames", config.MAX_FRAMES)
    add_optional_arg(cmd, "--eval-split-every", config.EVAL_SPLIT_EVERY)
    add_optional_arg(cmd, "--subsample-frames", config.SUBSAMPLE_FRAMES)
    add_optional_arg(cmd, "--subsample-points", config.SUBSAMPLE_POINTS)

    # Eval
    cmd.extend(["--eval-every", str(config.EVAL_EVERY)])
    if config.EVAL_SAVE_TO_DISK:
        cmd.append("--eval-save-to-disk")

    # 匯出
    export_every = (
        config.TOTAL_STEPS
        if config.EXPORT_EVERY is None
        else int(config.EXPORT_EVERY)
    )
    cmd.extend(["--export-every", str(export_every)])
    cmd.extend(["--export-path", str(output_dir)])

    # 預設只輸出最後一份時，直接叫 1.ply / 2.ply / ...
    # 若使用週期性 checkpoint，加入 {iter} 避免覆蓋。
    if config.EXPORT_EVERY is None:
        export_name = f"{group_name}.ply"
    else:
        export_name = f"{group_name}_{{iter}}.ply"

    cmd.extend(["--export-name", export_name])

    # 使用者自訂額外參數最後加入，可覆蓋/補充未單獨列出的設定。
    cmd.extend(str(x) for x in config.EXTRA_ARGS)

    return cmd


def run_brush(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
) -> int:
    """
    啟動 Brush，stdout/stderr 即時顯示並同步寫入 log。
    訓練本身完全不設定 timeout。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(quote_cmd(cmd))
        log_file.write("\n\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return process.wait()


def main() -> int:
    base_dir = config.BASE_DIR.resolve()
    input_dir = config.INPUT_DIR.resolve()
    output_root = config.OUTPUT_DIR.resolve()
    brush_exe = config.BRUSH_EXE

    print("=" * 72)
    print("Brush RGBA 高品質批次訓練")
    print("=" * 72)
    print(f"[程式目錄] {base_dir}")
    print(f"[Input]    {input_dir}")
    print(f"[Output]   {output_root}")
    print(f"[Brush]    {brush_exe}")
    print(f"[Alpha]    {config.ALPHA_MODE} / weight={config.MATCH_ALPHA_WEIGHT}")
    print(f"[MinScale] {config.MIN_SCALE_FACTOR}")
    print(f"[Growth]   stop @ {config.GROWTH_STOP_ITER}")
    print()

    if not brush_exe.is_file():
        print(f"[錯誤] 找不到 Brush 執行檔：{brush_exe}")
        return 1

    if not input_dir.is_dir():
        print(f"[錯誤] 找不到 input 資料夾：{input_dir}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    groups = sorted(
        [p for p in input_dir.iterdir() if p.is_dir()],
        key=lambda p: natural_key(p.name),
    )

    if not groups:
        print("[錯誤] input 裡沒有任何資料夾。")
        return 1

    help_text = get_brush_help(brush_exe)
    steps_flag = detect_steps_flag(help_text)

    flags_ok, flags_reason = validate_brush_flags(help_text)
    if not flags_ok:
        print(f"[錯誤] {flags_reason}")
        return 1

    print(f"[Brush steps 參數] {steps_flag}")
    print("[Brush 新流程參數] OK")
    print(f"[找到組數] {len(groups)}")
    print()

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in config.EXTRA_ENV.items()})

    success: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []

    for index, group_dir in enumerate(groups, start=1):
        group_name = group_dir.name
        dataset_dir = group_dir / config.DATASET_SUBDIR
        group_output_dir = output_root / group_name
        final_ply = group_output_dir / f"{group_name}.ply"
        log_path = group_output_dir / "brush.log"

        print()
        print("=" * 72)
        print(f"[{index}/{len(groups)}] {group_name}")
        print("=" * 72)
        print(f"[Dataset] {dataset_dir}")
        print(f"[Output]  {group_output_dir}")

        valid, reason = validate_dataset(dataset_dir)
        if not valid:
            print(f"[失敗] {reason}")
            failed.append((group_name, reason))
            if config.CONTINUE_ON_ERROR:
                continue
            return 1

        if (
            config.SKIP_COMPLETED
            and config.EXPORT_EVERY is None
            and final_ply.is_file()
            and final_ply.stat().st_size > 0
        ):
            print(f"[跳過] 已存在完成檔：{final_ply}")
            skipped.append(group_name)
            continue

        group_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(
            brush_exe=brush_exe,
            dataset_dir=dataset_dir.resolve(),
            group_name=group_name,
            output_dir=group_output_dir.resolve(),
            steps_flag=steps_flag,
        )

        print("[命令]")
        print(quote_cmd(cmd))
        print()

        if config.DRY_RUN:
            print("[DRY RUN] 未實際啟動 Brush。")
            skipped.append(group_name)
            continue

        try:
            return_code = run_brush(cmd, log_path, env)
        except KeyboardInterrupt:
            print("\n[中止] 使用者中止執行。")
            return 130
        except Exception as exc:
            reason = f"啟動 Brush 時發生例外：{exc}"
            print(f"[失敗] {reason}")
            failed.append((group_name, reason))
            if config.CONTINUE_ON_ERROR:
                continue
            return 1

        if return_code != 0:
            reason = f"Brush 結束碼 = {return_code}"
            print(f"[失敗] {reason}")
            print(f"[Log] {log_path}")
            failed.append((group_name, reason))
            if config.CONTINUE_ON_ERROR:
                continue
            return return_code or 1

        if config.EXPORT_EVERY is None:
            if final_ply.is_file() and final_ply.stat().st_size > 0:
                print(f"[完成] {final_ply}")
                success.append(group_name)
            else:
                ply_files = sorted(group_output_dir.glob("*.ply"))
                if ply_files:
                    reason = (
                        f"Brush 返回成功，但沒有找到預期的 {final_ply.name}；"
                        f"實際找到：{', '.join(p.name for p in ply_files)}"
                    )
                else:
                    reason = "Brush 返回成功，但輸出資料夾內沒有任何 .ply"

                print(f"[失敗] {reason}")
                failed.append((group_name, reason))
                if not config.CONTINUE_ON_ERROR:
                    return 1
        else:
            ply_files = sorted(group_output_dir.glob("*.ply"))
            if ply_files:
                print(f"[完成] 產生 {len(ply_files)} 個 PLY checkpoint")
                success.append(group_name)
            else:
                reason = "Brush 返回成功，但沒有產生任何 .ply"
                print(f"[失敗] {reason}")
                failed.append((group_name, reason))
                if not config.CONTINUE_ON_ERROR:
                    return 1

    print()
    print("=" * 72)
    print("批次處理結果")
    print("=" * 72)
    print(f"成功：{len(success)}")
    print(f"跳過：{len(skipped)}")
    print(f"失敗：{len(failed)}")

    if success:
        print("成功項目：" + ", ".join(success))

    if skipped:
        print("跳過項目：" + ", ".join(skipped))

    if failed:
        print("失敗項目：")
        for name, reason in failed:
            print(f"  - {name}: {reason}")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中止] 使用者中止執行。")
        sys.exit(130)
