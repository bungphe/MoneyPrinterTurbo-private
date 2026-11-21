import glob
import os
import pathlib
import shutil
from typing import Union

from fastapi import BackgroundTasks, Depends, Path, Request, UploadFile
from fastapi.params import File
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

from app.config import config
from app.controllers import base
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models import const
from app.models.schema import (
    AudioRequest,
    BatchVideoRequest,
    MergeVideoRequest,
    BgmRetrieveResponse,
    BgmUploadResponse,
    SubtitleRequest,
    TaskDeletionResponse,
    TaskQueryRequest,
    TaskQueryResponse,
    TaskResponse,
    TaskVideoRequest,
    VideoMaterialUploadResponse,
    VideoMaterialRetrieveResponse
)
from app.services import batch as batch_service
from app.services import state as sm
from app.services import task as tm
from app.utils import utils

# 认证依赖项
# router = new_router(dependencies=[Depends(base.verify_token)])
router = new_router()

_enable_redis = config.app.get("enable_redis", False)
_redis_host = config.app.get("redis_host", "localhost")
_redis_port = config.app.get("redis_port", 6379)
_redis_db = config.app.get("redis_db", 0)
_redis_password = config.app.get("redis_password", None)
_max_concurrent_tasks = config.app.get("max_concurrent_tasks", 5)

redis_url = f"redis://:{_redis_password}@{_redis_host}:{_redis_port}/{_redis_db}"
# 根据配置选择合适的任务管理器
if _enable_redis:
    task_manager = RedisTaskManager(
        max_concurrent_tasks=_max_concurrent_tasks, redis_url=redis_url
    )
else:
    task_manager = InMemoryTaskManager(max_concurrent_tasks=_max_concurrent_tasks)


def _sanitize_upload_filename(filename: str, request_id: str) -> str:
    # 浏览器或客户端有时会附带目录信息，甚至可能夹带 ../ 这类穿越片段。
    # 这里只保留纯文件名，避免上传接口把文件写到目标目录之外。
    normalized_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if not normalized_name or normalized_name in {".", ".."}:
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: invalid filename",
        )
    return normalized_name


def _resolve_path_within_directory(base_dir: str, unsafe_path: str, request_id: str) -> str:
    # 对用户传入的相对路径做归一化，并强制要求结果仍然落在指定目录内，
    # 这样可以阻止通过 ../ 或绝对路径逃逸到任务目录之外读取任意文件。
    base_dir_real = os.path.realpath(base_dir)
    resolved_path = os.path.realpath(os.path.join(base_dir_real, unsafe_path))

    try:
        common_path = os.path.commonpath([base_dir_real, resolved_path])
    except ValueError:
        raise HttpException(
            task_id=request_id,
            status_code=403,
            message=f"{request_id}: invalid file path",
        )

    if common_path != base_dir_real:
        raise HttpException(
            task_id=request_id,
            status_code=403,
            message=f"{request_id}: access to the requested file is forbidden",
        )

    if not os.path.isfile(resolved_path):
        raise HttpException(
            task_id=request_id,
            status_code=404,
            message=f"{request_id}: file not found",
        )

    return resolved_path


@router.post("/videos", response_model=TaskResponse, summary="Generate a short video")
def create_video(
    background_tasks: BackgroundTasks, request: Request, body: TaskVideoRequest
):
    return create_task(request, body, stop_at="video")


@router.post("/subtitle", response_model=TaskResponse, summary="Generate subtitle only")
def create_subtitle(
    background_tasks: BackgroundTasks, request: Request, body: SubtitleRequest
):
    return create_task(request, body, stop_at="subtitle")


@router.post("/audio", response_model=TaskResponse, summary="Generate audio only")
def create_audio(
    background_tasks: BackgroundTasks, request: Request, body: AudioRequest
):
    return create_task(request, body, stop_at="audio")


def create_task(
    request: Request,
    body: Union[TaskVideoRequest, SubtitleRequest, AudioRequest],
    stop_at: str,
):
    task_id = utils.get_uuid()
    request_id = base.get_task_id(request)
    try:
        task = {
            "task_id": task_id,
            "request_id": request_id,
            "params": body.model_dump(),
        }
        sm.state.update_task(task_id)
        task_manager.add_task(tm.start, task_id=task_id, params=body, stop_at=stop_at)
        logger.success(f"Task created: {utils.to_json(task)}")
        return utils.get_response(200, task)
    except ValueError as e:
        raise HttpException(
            task_id=task_id, status_code=400, message=f"{request_id}: {str(e)}"
        )



@router.post(
    "/videos/batch",
    response_model=TaskResponse,
    summary="Generate multiple videos from a list of subjects (optionally merge them)",
)
def create_batch_videos(request: Request, body: BatchVideoRequest):
    task_id = utils.get_uuid()
    request_id = base.get_task_id(request)
    base_params = body.settings or TaskVideoRequest(video_subject="")
    items = [
        batch_service.BatchItem(
            subject=item.video_subject.strip(),
            script=item.video_script or "",
            keywords=item.video_terms or "",
            style=item.video_script_style or "",
        )
        for item in body.items
        if item.video_subject and item.video_subject.strip()
    ]
    if not items:
        raise HttpException(
            task_id=task_id, status_code=400, message=f"{request_id}: no valid subjects"
        )
    sm.state.update_task(task_id, total=len(items))
    task_manager.add_task(
        batch_service.run_batch_task,
        task_id=task_id,
        items=items,
        base_params=base_params,
        merge_videos=body.merge_videos,
        merge_transition=body.merge_transition,
    )
    logger.success(f"Batch task created: {task_id}, {len(items)} videos")
    return utils.get_response(200, {"task_id": task_id})


def _resolve_merge_input(video_file: str, request_id: str) -> str:
    # 只允许合并任务输出（storage/tasks）和已上传素材（storage/local_videos）中的文件，
    # 防止通过该接口读取服务器上的任意文件
    path = (video_file or "").strip()
    if "://" in path:
        # 支持直接传入 GET /tasks/{task_id} 返回的视频地址
        marker = "/tasks/"
        if marker not in path:
            raise HttpException(
                task_id=request_id, status_code=400, message=f"{request_id}: invalid video url"
            )
        path = os.path.join("tasks", path.split(marker, 1)[1])
    storage_dir = utils.storage_dir()
    if not os.path.isabs(path):
        path = os.path.join(storage_dir, path)
    for allowed_dir in (utils.task_dir(), utils.storage_dir("local_videos", create=True)):
        allowed_real = os.path.realpath(allowed_dir)
        try:
            inside = os.path.commonpath([allowed_real, os.path.realpath(path)]) == allowed_real
        except ValueError:
            # Windows 上不同盘符的路径无法比较，视为不在允许目录内
            inside = False
        if inside:
            return _resolve_path_within_directory(
                allowed_dir, os.path.relpath(os.path.realpath(path), allowed_real), request_id
            )
    raise HttpException(
        task_id=request_id,
        status_code=403,
        message=f"{request_id}: only files in storage/tasks or storage/local_videos can be merged",
    )


@router.post(
    "/videos/merge",
    response_model=TaskResponse,
    summary="Merge several generated videos into one (straight cut or crossfade)",
)
def merge_videos(request: Request, body: MergeVideoRequest):
    request_id = base.get_task_id(request)
    files = []
    for source_task_id in body.task_ids or []:
        task = sm.state.get_task(source_task_id)
        if not task or task.get("state") != const.TASK_STATE_COMPLETE or not task.get("videos"):
            raise HttpException(
                task_id=request_id,
                status_code=400,
                message=f"{request_id}: task {source_task_id} not found or has no finished videos",
            )
        files.extend(_resolve_merge_input(v, request_id) for v in task["videos"])
    for video_file in body.video_files or []:
        files.append(_resolve_merge_input(video_file, request_id))
    if len(files) < 2:
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: at least two videos are required",
        )

    task_id = utils.get_uuid()
    sm.state.update_task(task_id)
    task_manager.add_task(
        batch_service.run_merge_task, task_id=task_id, files=files, transition=body.transition
    )
    logger.success(f"Merge task created: {task_id}, {len(files)} videos")
    return utils.get_response(200, {"task_id": task_id})


from fastapi import Query

@router.get("/tasks", response_model=TaskQueryResponse, summary="Get all tasks")
def get_all_tasks(request: Request, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    request_id = base.get_task_id(request)
    tasks, total = sm.state.get_all_tasks(page, page_size)

    response = {
        "tasks": tasks,
        "total": total,
        "page": page,
        "page_size": page_size,
    }
    return utils.get_response(200, response)



@router.get(
    "/tasks/{task_id}", response_model=TaskQueryResponse, summary="Query task status"
)
def get_task(
    request: Request,
    task_id: str = Path(..., description="Task ID"),
    query: TaskQueryRequest = Depends(),
):
    endpoint = config.app.get("endpoint", "")
    if not endpoint:
        endpoint = str(request.base_url)
    endpoint = endpoint.rstrip("/")

    request_id = base.get_task_id(request)
    task = sm.state.get_task(task_id)
    if task:
        # 复制一份再转换为 URL，避免内存状态中保存的文件路径被改写成 URL
        task = dict(task)
        task_dir = utils.task_dir()

        def file_to_uri(file):
            if not file.startswith(endpoint):
                _uri_path = file.replace(task_dir, "tasks").replace("\\", "/")
                _uri_path = f"{endpoint}/{_uri_path}"
            else:
                _uri_path = file
            return _uri_path

        if "videos" in task:
            videos = task["videos"]
            urls = []
            for v in videos:
                urls.append(file_to_uri(v))
            task["videos"] = urls
        if "combined_videos" in task:
            combined_videos = task["combined_videos"]
            urls = []
            for v in combined_videos:
                urls.append(file_to_uri(v))
            task["combined_videos"] = urls
        # 批量生成 / 合并任务的结果文件
        for key in ("merged_video", "csv_file"):
            if task.get(key):
                task[key] = file_to_uri(task[key])
        if isinstance(task.get("results"), list):
            task["results"] = [
                {**r, "videos": [file_to_uri(v) for v in r.get("videos", [])]}
                if isinstance(r, dict)
                else r
                for r in task["results"]
            ]
        return utils.get_response(200, task)

    raise HttpException(
        task_id=task_id, status_code=404, message=f"{request_id}: task not found"
    )


@router.delete(
    "/tasks/{task_id}",
    response_model=TaskDeletionResponse,
    summary="Delete a generated short video task",
)
def delete_video(request: Request, task_id: str = Path(..., description="Task ID")):
    request_id = base.get_task_id(request)
    task = sm.state.get_task(task_id)
    if task:
        tasks_dir = utils.task_dir()
        current_task_dir = os.path.join(tasks_dir, task_id)
        if os.path.exists(current_task_dir):
            shutil.rmtree(current_task_dir)

        sm.state.delete_task(task_id)
        logger.success(f"video deleted: {utils.to_json(task)}")
        return utils.get_response(200)

    raise HttpException(
        task_id=task_id, status_code=404, message=f"{request_id}: task not found"
    )


@router.get(
    "/musics", response_model=BgmRetrieveResponse, summary="Retrieve local BGM files"
)
def get_bgm_list(request: Request):
    suffix = "*.mp3"
    song_dir = utils.song_dir()
    files = glob.glob(os.path.join(song_dir, suffix))
    bgm_list = []
    for file in files:
        bgm_list.append(
            {
                "name": os.path.basename(file),
                "size": os.path.getsize(file),
                "file": file,
            }
        )
    response = {"files": bgm_list}
    return utils.get_response(200, response)


@router.post(
    "/musics",
    response_model=BgmUploadResponse,
    summary="Upload the BGM file to the songs directory",
)
def upload_bgm_file(request: Request, file: UploadFile = File(...)):
    request_id = base.get_task_id(request)
    safe_filename = _sanitize_upload_filename(file.filename, request_id)
    # check file ext
    if safe_filename.lower().endswith("mp3"):
        song_dir = utils.song_dir()
        save_path = os.path.join(song_dir, safe_filename)
        # save file
        with open(save_path, "wb+") as buffer:
            # If the file already exists, it will be overwritten
            file.file.seek(0)
            buffer.write(file.file.read())
        response = {"file": save_path}
        return utils.get_response(200, response)

    raise HttpException(
        "", status_code=400, message=f"{request_id}: Only *.mp3 files can be uploaded"
    )

@router.get(
    "/video_materials", response_model=VideoMaterialRetrieveResponse, summary="Retrieve local video materials"
)
def get_video_materials_list(request: Request):
    allowed_suffixes = ("mp4", "mov", "avi", "flv", "mkv", "jpg", "jpeg", "png")
    local_videos_dir = utils.storage_dir("local_videos", create=True)
    files = []
    for suffix in allowed_suffixes:
        files.extend(glob.glob(os.path.join(local_videos_dir, f"*.{suffix}")))
    # 文件系统枚举顺序不稳定，直接返回会导致“顺序拼接”在不同机器或不同
    # 时刻表现不一致。这里统一按文件名排序，至少保证服务端返回顺序可预测。
    files.sort(key=lambda file_path: os.path.basename(file_path).lower())
    video_materials_list = []
    for file in files:
        video_materials_list.append(
            {
                "name": os.path.basename(file),
                "size": os.path.getsize(file),
                "file": file,
            }
        )
    response = {"files": video_materials_list}
    return utils.get_response(200, response)


@router.post(
    "/video_materials",
    response_model=VideoMaterialUploadResponse,
    summary="Upload the video material file to the local videos directory",
)
def upload_video_material_file(request: Request, file: UploadFile = File(...)):
    request_id = base.get_task_id(request)
    safe_filename = _sanitize_upload_filename(file.filename, request_id)
    # check file ext
    allowed_suffixes = ("mp4", "mov", "avi", "flv", "mkv", "jpg", "jpeg", "png")
    normalized_filename = safe_filename.lower()
    # 统一按小写扩展名校验，兼容 .MOV 这类大写后缀文件。
    if normalized_filename.endswith(allowed_suffixes):
        local_videos_dir = utils.storage_dir("local_videos", create=True)
        save_path = os.path.join(local_videos_dir, safe_filename)
        # save file
        with open(save_path, "wb+") as buffer:
            # If the file already exists, it will be overwritten
            file.file.seek(0)
            buffer.write(file.file.read())
        response = {"file": save_path}
        return utils.get_response(200, response)

    raise HttpException(
        "", status_code=400, message=f"{request_id}: Only files with extensions {', '.join(allowed_suffixes)} can be uploaded"
    )

@router.get("/stream/{file_path:path}")
async def stream_video(request: Request, file_path: str):
    request_id = base.get_task_id(request)
    tasks_dir = utils.task_dir()
    video_path = _resolve_path_within_directory(tasks_dir, file_path, request_id)
    range_header = request.headers.get("Range")
    video_size = os.path.getsize(video_path)
    start, end = 0, video_size - 1

    length = video_size
    if range_header:
        range_ = range_header.split("bytes=")[1]
        start, end = [int(part) if part else None for part in range_.split("-")]
        if start is None:
            start = video_size - end
            end = video_size - 1
        if end is None:
            end = video_size - 1
        length = end - start + 1

    def file_iterator(file_path, offset=0, bytes_to_read=None):
        with open(file_path, "rb") as f:
            f.seek(offset, os.SEEK_SET)
            remaining = bytes_to_read or video_size
            while remaining > 0:
                bytes_to_read = min(4096, remaining)
                data = f.read(bytes_to_read)
                if not data:
                    break
                remaining -= len(data)
                yield data

    response = StreamingResponse(
        file_iterator(video_path, start, length), media_type="video/mp4"
    )
    response.headers["Content-Range"] = f"bytes {start}-{end}/{video_size}"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Length"] = str(length)
    response.status_code = 206  # Partial Content

    return response


@router.get("/download/{file_path:path}")
async def download_video(request: Request, file_path: str):
    """
    download video
    :param request: Request request
    :param file_path: video file path, eg: /cd1727ed-3473-42a2-a7da-4faafafec72b/final-1.mp4
    :return: video file
    """
    request_id = base.get_task_id(request)
    tasks_dir = utils.task_dir()
    video_path = _resolve_path_within_directory(tasks_dir, file_path, request_id)
    file_path = pathlib.Path(video_path)
    filename = file_path.stem
    extension = file_path.suffix
    headers = {"Content-Disposition": f"attachment; filename={filename}{extension}"}
    return FileResponse(
        path=video_path,
        headers=headers,
        filename=f"{filename}{extension}",
        media_type=f"video/{extension[1:]}",
    )
