from __future__ import annotations

import math
import struct
import zlib
from io import BytesIO
from typing import Any

_NIS_CUSTOM_DATA_TAG = 65330
_NIS_PICTURE_METADATA_TAG = 65331
_NIS_APP_DATA_TAG = 65332
_NIS_CALIBRATION_TAG = 65326
_NIS_PRIVATE_TAGS = frozenset(range(65325, 65334))

_CLX_BOOL = 1
_CLX_INT32 = 2
_CLX_UINT32 = 3
_CLX_INT64 = 4
_CLX_UINT64 = 5
_CLX_DOUBLE = 6
_CLX_POINTER = 7
_CLX_STRING = 8
_CLX_BYTEARRAY = 9
_CLX_LEVEL = 11
_CLX_COMPRESSED = 76


class _InvalidNisMetadata(ValueError):
    pass


def _read_exact(stream: BytesIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise _InvalidNisMetadata("truncated NIS-Elements metadata")
    return value


def _read_struct(stream: BytesIO, fmt: str):
    parser = struct.Struct(fmt)
    return parser.unpack(_read_exact(stream, parser.size))


def _read_clx_string(stream: BytesIO) -> str:
    chunks = bytearray()
    while True:
        pair = _read_exact(stream, 2)
        if pair == b"\x00\x00":
            break
        chunks.extend(pair)
    return bytes(chunks).decode("utf-16le", errors="replace")


def _read_clx_name(stream: BytesIO) -> tuple[str, int]:
    header = stream.read(2)
    if not header:
        return "", -1
    if len(header) != 2:
        raise _InvalidNisMetadata("truncated NIS-Elements item header")
    data_type, name_length = struct.unpack("BB", header)
    if data_type in {0, 10}:
        raise _InvalidNisMetadata(f"unsupported NIS metadata type {data_type}")
    if data_type == _CLX_COMPRESSED:
        return "", data_type
    raw_name = _read_exact(stream, name_length * 2)
    name = raw_name.decode("utf-16le", errors="replace").rstrip("\x00")
    return name, data_type


def _read_clx_scalar(stream: BytesIO, data_type: int):
    if data_type == _CLX_BOOL:
        return bool(_read_struct(stream, "<B")[0])
    if data_type == _CLX_INT32:
        return int(_read_struct(stream, "<i")[0])
    if data_type == _CLX_UINT32:
        return int(_read_struct(stream, "<I")[0])
    if data_type == _CLX_INT64:
        return int(_read_struct(stream, "<q")[0])
    if data_type in {_CLX_UINT64, _CLX_POINTER}:
        return int(_read_struct(stream, "<Q")[0])
    if data_type == _CLX_DOUBLE:
        return float(_read_struct(stream, "<d")[0])
    if data_type == _CLX_STRING:
        return _read_clx_string(stream)
    if data_type == _CLX_BYTEARRAY:
        size = int(_read_struct(stream, "<Q")[0])
        return _read_exact(stream, size)
    raise _InvalidNisMetadata(f"unsupported NIS metadata type {data_type}")


def _decode_clx_items(
    stream: BytesIO,
    count: int = 1,
    *,
    depth: int = 0,
) -> dict[str, Any]:
    if depth > 64:
        raise _InvalidNisMetadata("NIS metadata nesting is too deep")
    output: dict[str, Any] = {}
    for _ in range(count):
        item_start = stream.tell()
        name, data_type = _read_clx_name(stream)
        if data_type == -1:
            break
        if data_type == _CLX_COMPRESSED:
            _read_exact(stream, 10)
            try:
                inflated = zlib.decompress(stream.read())
            except zlib.error as error:
                raise _InvalidNisMetadata("invalid compressed NIS metadata") from error
            return _decode_clx_items(BytesIO(inflated), depth=depth + 1)
        if data_type == _CLX_LEVEL:
            item_count, item_length = _read_struct(stream, "<IQ")
            consumed = stream.tell() - item_start
            child_length = int(item_length) - consumed
            if child_length < 0:
                raise _InvalidNisMetadata("invalid NIS metadata level length")
            child_data = _read_exact(stream, child_length)
            value: Any = _decode_clx_items(
                BytesIO(child_data), int(item_count), depth=depth + 1
            )
            _read_exact(stream, int(item_count) * 8)
        else:
            value = _read_clx_scalar(stream, data_type)

        if name == "" and name in output:
            previous = output[name]
            output[name] = previous + [value] if isinstance(previous, list) else [previous, value]
        else:
            output[name] = value
    return output


def _find_clx_level(blob: bytes, name: str) -> dict[str, Any]:
    encoded_name = (name + "\x00").encode("utf-16le")
    marker = bytes((_CLX_LEVEL, len(name) + 1)) + encoded_name
    offset = blob.find(marker)
    if offset < 0:
        return {}
    try:
        return _decode_clx_items(BytesIO(blob[offset:]))
    except (OSError, OverflowError, struct.error, _InvalidNisMetadata):
        return {}


def _directory_entries(blob: bytes) -> dict[str, bytes]:
    """Read the simple name/size/offset directory used by NIS TIFF tag 65330."""
    try:
        count = struct.unpack_from("<I", blob, 0)[0]
        if count > 1024:
            return {}
        position = 4
        records: list[tuple[str, int, int]] = []
        for _ in range(count):
            name_length = struct.unpack_from("<I", blob, position)[0]
            position += 4
            if name_length > 4096:
                return {}
            name_size = int(name_length) * 2
            name = blob[position : position + name_size].decode("utf-16le")
            position += name_size
            size, offset = struct.unpack_from("<II", blob, position)
            position += 8
            end = int(offset) + int(size)
            if offset < position or end > len(blob):
                return {}
            records.append((name, int(offset), end))
    except (UnicodeDecodeError, struct.error):
        return {}
    return {name: blob[start:end] for name, start, end in records}


def _tag_value(page, code: int):
    tag = page.tags.get(code)
    return getattr(tag, "value", None)


def _positive_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _mapping(value) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_mapping(value) -> dict[str, Any]:
    mapping = _mapping(value)
    return next((item for item in mapping.values() if isinstance(item, dict)), {})


def _picture_metadata(page) -> dict[str, Any]:
    value = _tag_value(page, _NIS_PICTURE_METADATA_TAG)
    if not isinstance(value, bytes):
        return {}
    decoded = _find_clx_level(value, "SLxPictureMetadata")
    return _mapping(decoded.get("SLxPictureMetadata"))


def _text_metadata(page) -> dict[str, Any]:
    value = _tag_value(page, _NIS_CUSTOM_DATA_TAG)
    if not isinstance(value, bytes):
        return {}
    chunk = _directory_entries(value).get("TextInfoTiffV1_0")
    if not chunk:
        return {}
    decoded = _find_clx_level(chunk, "SLxImageTextInfo")
    return _mapping(decoded.get("SLxImageTextInfo"))


def read_nis_elements_tiff_metadata(page) -> tuple[float | None, dict[str, Any]]:
    """Return XY calibration and a compact metadata summary for NIS TIFF files."""
    if not any(page.tags.get(code) is not None for code in _NIS_PRIVATE_TAGS):
        return None, {}
    picture = _picture_metadata(page)
    text = _text_metadata(page)
    if not picture and _tag_value(page, _NIS_APP_DATA_TAG) is None:
        return None, {}

    calibration = _positive_float(_tag_value(page, _NIS_CALIBRATION_TAG))
    if calibration is None:
        calibration = _positive_float(picture.get("dCalibration"))

    extras: dict[str, Any] = {
        "nis_elements": True,
        "nis_metadata_format": "NIS-Elements TIFF private tags 65325-65333",
        "nis_raw_metadata_available": True,
    }
    if calibration is not None:
        extras["nis_calibration_um"] = calibration

    stage_values = tuple(
        picture.get(key) for key in ("dXPos", "dYPos", "dZPos")
    )
    if all(isinstance(value, int | float) for value in stage_values):
        extras["stage_position_um"] = tuple(float(value) for value in stage_values)

    scalar_fields = {
        "dTimeMSec": "nis_relative_time_ms",
        "dTimeAbsolute": "nis_absolute_julian_day",
        "wsObjectiveName": "objective_name",
        "dObjectiveNA": "objective_na",
        "dRefractIndex1": "refractive_index",
        "dZoom": "zoom_magnification",
    }
    for source, destination in scalar_fields.items():
        value = picture.get(source)
        if value not in (None, "", -1, -1.0):
            extras[destination] = value

    planes = _mapping(picture.get("sPicturePlanes"))
    sample = _first_mapping(planes.get("sSampleSetting"))
    camera = _mapping(sample.get("pCameraSetting"))
    camera_name = camera.get("CameraUserName") or camera.get("CameraUniqueName")
    if camera_name:
        extras["camera_name"] = camera_name
    exposure = _positive_float(_mapping(camera.get("PropertiesFast")).get("Exposure"))
    if exposure is not None:
        extras["exposure_ms"] = exposure

    objective = _mapping(sample.get("pObjectiveSetting"))
    objective_fields = {
        "wsObjectiveName": "objective_name",
        "dObjectiveMag": "objective_magnification",
        "dObjectiveNA": "objective_na",
        "dRefractIndex": "refractive_index",
    }
    for source, destination in objective_fields.items():
        value = objective.get(source)
        if value not in (None, "", -1, -1.0):
            extras[destination] = value

    optical_configs = _mapping(sample.get("sOpticalConfigs"))
    optical_names = [
        item.get("sOpticalConfigName")
        for item in optical_configs.values()
        if isinstance(item, dict) and item.get("sOpticalConfigName")
    ]
    if optical_names:
        extras["optical_config_names"] = optical_names

    text_fields = {
        "TextInfoItem_5": "nis_description",
        "TextInfoItem_6": "nis_capturing",
        "TextInfoItem_9": "acquisition_datetime",
        "TextInfoItem_13": "nis_optics",
    }
    for source, destination in text_fields.items():
        value = text.get(source)
        if value:
            extras[destination] = value
    return calibration, extras
