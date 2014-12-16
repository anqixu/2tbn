function probs = sigmoidCPD(xprime, x, d, kd, od, bd)
% Computes Prob(d={1, 0, -1}|...) using Sigmoid(...; kd, od, bd)
%
% More specifically:
% - Prob(d==+1|...) = bd + (1-3*bd) / [ 1 + exp(- kd[ (x-xprime) - od] ) ]
% - Prob(d==-1|...) = bd + (1-3*bd) / [ 1 + exp(- kd[-(x-xprime) - od] ) ]
% - Prob(d== 0|...) = 1 - Prob(d==+1) - Prob(d==-1)
%
% - d: observation of direction of motion/change, d \in {-1, 0, 1}
%     (e.g. d = sign(x - xprime))
% - kd: sigmoid gain; controls the steepness of the sigmoid slope
% - od: center offset; defines point on (x-xprime) space where prob==0.5
% - bd: bias probability; defines probability of erroneously
%       signaling d, regardless of actual value of (x-xprime)
%
% - outputs valid PMFs for p(d=-1|...), p(d=0|...) and p(d=1|...) that
%   sum to 1, notwithstanding numerical precision issues

dx = x - xprime;
scale = 1-3*bd;

if d > 0,
  probs = bd + scale./(1+exp(- kd*(+dx-od) ));
elseif d < 0;
  probs = bd + scale./(1+exp(- kd*(-dx-od) ));
else % d == 0
  probs = 1 - 2*bd - scale.*(1./(1+exp(- kd*(+dx-od) )) + 1./(1+exp(- kd*(-dx-od) )));
  probs(probs < 0) = 0; % Address numerical issues
end

end
