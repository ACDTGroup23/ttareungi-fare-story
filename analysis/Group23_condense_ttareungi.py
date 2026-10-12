# =====================================================================================
# Group 23 · 따릉이 시간대별 이용정보 → 작은 요약 파일 1개 (station_day_cache.csv.gz)
#
# 12개 월별 원본(각 200–450 MB)을 읽어서, 우리 비교 기간의 평일만 남기고
# "날짜 × 대여소"별 전체 대여 수와 출퇴근 시간(07–08시, 18–19시) 대여 수로 줄입니다.
# 결과 파일은 보통 10 MB 안팎이라 채팅에 바로 올릴 수 있어요.
#
# [Colab] 1) 원본 12개를 Google Drive의 한 폴더에 올리기 (예: 내 드라이브/ttareungi/hourly)
#         2) 새 Colab 노트북 셀에 이 파일 내용을 통째로 붙여넣기
#         3) 아래 INPUT_DIR을 그 폴더로 바꾸고 실행 → 같은 폴더에 결과 파일 생성
# [내 컴퓨터] pandas가 있으면:  python Group23_condense_ttareungi.py "원본폴더경로"
# =====================================================================================
import os, sys, glob
import numpy as np
import pandas as pd

INPUT_DIR = "/content/drive/MyDrive/ttareungi/hourly"     # ← 원본 12개가 들어 있는 폴더
OUTPUT = None                                            # None이면 INPUT_DIR 안에 저장

# 우리 비교 기간 (브랜든 방식): 이 날짜 범위의 평일만 남깁니다
WINDOWS = [("2022-10-07", "2022-12-31"), ("2023-10-07", "2023-12-31"),
           ("2024-07-01", "2024-09-30"), ("2025-07-01", "2025-09-30")]
COMMUTE = [7, 8, 18, 19]
ENCODINGS = ("cp949", "utf-8-sig", "utf-8", "euc-kr")


def _pick(cols, *names):
    cols = [str(c).strip() for c in cols]
    for n in names:
        for c in cols:
            if n == c:
                return c
    for n in names:
        for c in cols:
            if n in c:
                return c
    raise KeyError(f"{names} 중 어느 것도 없음. 열 이름: {cols}")


def _norm_id(s):
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(~s.str.fullmatch(r"\d+"), s.str.lstrip("0").replace("", "0"))


def _in_windows(d):
    keep = pd.Series(False, index=d.index)
    for a, b in WINDOWS:
        keep |= (d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))
    return keep & (d.dt.weekday < 5)


class WrongDataset(Exception):
    pass


def _columns(path, head):
    cols = [str(c).strip() for c in head.columns]
    has = lambda *ns: any(n in c for c in cols for n in ns)
    if has("대여년월") and not has("대여일자", "대여일시", "대여시간"):
        raise WrongDataset(
            f"\n[{os.path.basename(path)}] 이 파일은 '이용정보(월별)' 데이터예요 (첫 열: 대여년월).\n"
            "날짜와 시간이 없어서 평일만 고르거나 출퇴근 시간 비율을 계산할 수 없어요.\n"
            "서울 열린데이터광장의 '서울시 공공자전거 이용정보(시간대별)' (OA-15245) 파일을 받아 주세요.\n"
            "그 파일에는 '대여일자'와 '대여시간'(0~23) 열이 있어요.")
    date = _pick(cols, "대여일자", "대여일시", "대여날짜")
    try:
        hour = _pick(cols, "대여시간", "시간대")
    except KeyError:
        hour = None                      # '2022-10-07 08:00'처럼 날짜 열에 시간이 붙어 있는 경우
    try:
        n = _pick(cols, "이용건수", "대여건수")
    except KeyError:
        n = None                         # 건수 열이 없으면 한 행 = 대여 1건으로 셉니다
    return {"date": date, "hour": hour, "station": _pick(cols, "대여소번호", "대여소 번호"), "n": n}


def _sniff_encoding(path):
    """Pick the encoding that decodes the first 1 MB cleanly."""
    raw = open(path, "rb").read(1_000_000)
    for enc in ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError as err:
            if err.start > len(raw) - 4:          # only the last, cut-off character failed
                return enc
    return "cp949"


def _parse_dates(raw):
    """Handles 2023-12-01, 20231201, 2023.12.01, 2023/12/01, 12/1/2023, '2023-12-01 08:00' and Excel serial numbers."""
    s = raw.astype(str).str.strip().str.replace('"', "", regex=False)
    out = pd.to_datetime(s.str[:10].str.replace(r"[./]", "-", regex=True), errors="coerce", format="%Y-%m-%d")
    left = out.isna()
    if left.any():
        digits = s[left].str.fullmatch(r"\d{8}")
        out[left & digits.reindex(s.index, fill_value=False)] = pd.to_datetime(
            s[left & digits.reindex(s.index, fill_value=False)], format="%Y%m%d", errors="coerce")
        left = out.isna()
    if left.any():
        num = pd.to_numeric(s[left], errors="coerce")
        serial = num.between(40000, 50000)                    # Excel saved the date as a number
        out[left & serial.reindex(s.index, fill_value=False)] = pd.to_datetime(
            num[serial], unit="D", origin="1899-12-30")
        left = out.isna()
    if left.any():
        out[left] = pd.to_datetime(s[left], errors="coerce")   # anything else pandas can guess (e.g. 12/1/2023 8:00)
    return out.dt.normalize()


def condense_file(path, chunksize=2_000_000):
    enc = _sniff_encoding(path)
    head = pd.read_csv(path, encoding=enc, nrows=5, encoding_errors="replace", on_bad_lines="skip")
    c = _columns(path, head)
    use = [v for v in c.values() if v is not None]
    parts, rows_read, unparsed, samples = [], 0, 0, []
    reader = pd.read_csv(path, encoding=enc, usecols=use, chunksize=chunksize, low_memory=False, dtype=str,
                         encoding_errors="replace", on_bad_lines="skip")   # one broken line no longer stops the file
    for ch in reader:
        rows_read += len(ch)
        raw_date = ch[c["date"]]
        date = _parse_dates(raw_date)
        unparsed += int(date.isna().sum())
        if len(samples) < 5:
            samples += raw_date.dropna().astype(str).head(5 - len(samples)).tolist()
        if c["hour"] is not None:
            hour = pd.to_numeric(ch[c["hour"]].str.strip(), errors="coerce")
        else:
            hour = pd.to_datetime(raw_date, errors="coerce").dt.hour
        keep = _in_windows(date)
        if not keep.any():
            continue
        n = (pd.to_numeric(ch.loc[keep, c["n"]].str.replace(",", ""), errors="coerce").fillna(0)
             if c["n"] is not None else pd.Series(1, index=ch.index[keep]))
        small = pd.DataFrame({"date": date[keep], "station": _norm_id(ch.loc[keep, c["station"]]),
                              "total": n, "commute": np.where(hour[keep].isin(COMMUTE), n, 0)})
        parts.append(small.groupby(["date", "station"], as_index=False)[["total", "commute"]].sum())
    out = pd.concat(parts) if parts else pd.DataFrame(columns=["date", "station", "total", "commute"])
    rng = f"{out.date.min():%Y-%m-%d} – {out.date.max():%Y-%m-%d}" if len(out) else "비교 기간 밖 (남긴 행 없음)"
    print(f"  {os.path.basename(path):40s} 원본 {rows_read:>10,}행 → {len(out):>8,}행 · {rng}  [인코딩 {enc}]")
    if unparsed or not len(out):
        print(f"    ⚠ 날짜를 못 읽은 행 {unparsed:,}개. 날짜 열 '{c['date']}'의 예: {samples}")
        print("      이 줄을 그대로 복사해서 채팅에 보내 주세요.")
    return out


def main(input_dir=INPUT_DIR, output=OUTPUT):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"{input_dir} 에 CSV가 없어요")
    print(f"{len(paths)}개 파일 처리 중...")
    try:
        sd = pd.concat([condense_file(p) for p in paths])
    except WrongDataset as e:
        print(e)
        return
    sd = sd.groupby(["date", "station"], as_index=False)[["total", "commute"]].sum()
    sd["date"] = pd.to_datetime(sd["date"]).dt.strftime("%Y-%m-%d")
    output = output or os.path.join(input_dir, "station_day_cache.csv.gz")
    sd.to_csv(output, index=False, compression="gzip")
    print(f"\n완료: {output}  ({os.path.getsize(output) / 1e6:.1f} MB, {len(sd):,}행)")
    check = sd.assign(year=sd.date.str[:4]).groupby("year").agg(
        days=("date", "nunique"), stations=("station", "nunique"),
        rentals=("total", "sum"), commute_share=("commute", "sum"))
    check["commute_share"] = (100 * check.commute_share / check.rentals).round(1)
    print("\n연도별 확인 (평일 수는 공휴일 포함 기준: 2022 61 · 2023 60 · 2024 66 · 2025 66 근처면 정상)")
    print(check.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else INPUT_DIR)
