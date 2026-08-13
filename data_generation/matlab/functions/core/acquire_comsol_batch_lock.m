%% acquire_comsol_batch_lock.m
% ============================================================
% Acquire one kernel-owned lease for a MATLAB batch producer.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Opens the requested lock file and acquires a non-blocking Java FileLock.
%   The lease prevents concurrent MATLAB producers from publishing the same
%   batch while allowing an interrupted batch to resume after process exit.
%
% USAGE
%   cleanup = acquire_comsol_batch_lock(lock_path)
%
% INPUTS
%   lock_path
%       Non-empty path to the persistent lock file. Its parent must exist.
%
% OUTPUTS
%   cleanup
%       onCleanup owner that releases the FileLock, channel, and file handle.
%
% NOTES
%   File contents are not the ownership signal. The operating system releases
%   the kernel lock when MATLAB exits, so no stale-sentinel deletion is needed.
% ============================================================

function cleanup = acquire_comsol_batch_lock(lock_path)

lock_path = require_text(lock_path);
lock_parent = fileparts(lock_path);
if isempty(lock_parent) || ~isfolder(lock_parent)
    error('acquire_comsol_batch_lock:ParentDirectory', ...
        'The batch lock parent directory does not exist: %s', ...
        lock_parent);
end
if isfolder(lock_path)
    error('acquire_comsol_batch_lock:LockPath', ...
        'The batch lock path is unexpectedly a directory: %s', ...
        lock_path);
end

random_access_file = java.io.RandomAccessFile(lock_path, 'rw');
channel = random_access_file.getChannel();
try
    file_lock = channel.tryLock();
catch lock_error
    close_resources(channel, random_access_file);
    if contains(string(lock_error.message), ...
            'OverlappingFileLockException')
        error('acquire_comsol_batch_lock:Unavailable', ...
            'Another MATLAB batch producer already owns: %s', ...
            lock_path);
    end
    rethrow(lock_error);
end
if isempty(file_lock)
    close_resources(channel, random_access_file);
    error('acquire_comsol_batch_lock:Unavailable', ...
        'Another MATLAB batch producer already owns: %s', lock_path);
end
cleanup = onCleanup(@() release_resources( ...
    file_lock, channel, random_access_file));
end

function release_resources(file_lock, channel, random_access_file)
try
    if file_lock.isValid()
        file_lock.release();
    end
catch
    % Cleanup must not hide the batch error that triggered unwinding.
end
close_resources(channel, random_access_file);
end

function close_resources(channel, random_access_file)
try
    channel.close();
catch
end
try
    random_access_file.close();
catch
end
end

function text = require_text(value)
if ischar(value) && isrow(value) && ~isempty(value)
    text = value;
elseif isstring(value) && isscalar(value) && ...
        ~ismissing(value) && strlength(value) > 0
    text = char(value);
else
    error('acquire_comsol_batch_lock:LockPath', ...
        'lock_path must be a non-empty text scalar.');
end
end
