%% kc_porosity_inverse.m
% ============================================================
% Invert the fixed Kozeny-Carman porosity response.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Solves target = eps^3 / (1 - eps)^2 for eps on the open interval
%   (0, 1) using a deterministic bounded bisection.
%
% USAGE
%   eps_value = kc_porosity_inverse(target)
%
% INPUTS
%   target
%       Finite, strictly positive scalar or numeric array inside the validated
%       response bracket.
%
% OUTPUTS
%   eps_value
%       Porosity values with the same shape as target and 0 < eps_value < 1.
%
% NOTES
%   The response is monotone on the validated interval. One hundred bisection
%   iterations provide deterministic inversion without external toolboxes.
% ============================================================

function eps_value = kc_porosity_inverse(target)

if any(~isfinite(target(:)) | target(:) <= 0)
    error('kc_porosity_inverse:InvalidTarget', ...
        'KC inversion target must be finite and strictly positive.');
end
lo = 1e-12;
hi = 1 - 1e-12;
response = @(value) value.^3 ./ (1 - value).^2;
if any(target(:) <= response(lo) | target(:) >= response(hi))
    error('kc_porosity_inverse:UnbracketedTarget', ...
        'KC inversion target is outside the validated open porosity bracket.');
end
eps_value = zeros(size(target));
for index = 1:numel(target)
    lower = lo;
    upper = hi;
    for iteration = 1:100
        midpoint = 0.5 * (lower + upper);
        if response(midpoint) > target(index)
            upper = midpoint;
        else
            lower = midpoint;
        end
    end
    eps_value(index) = 0.5 * (lower + upper);
end
end
