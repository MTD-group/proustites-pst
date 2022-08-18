(* ::Package:: *)

LaunchKernels[4]; (** Parallel threads **)

(**Tau range (range of spin lifetimes to compute determinant, in ps) **)
taurange = Range[1.1, 5, 0.01]
Export["tau_0.csv", taurange]

K = { {Om, I ayx qy/hbar tau, I axy qx/hbar tau, I ayz qy/hbar tau}, {I ayx qy/hbar tau, Om, 2 ayz ky/hbar tau, -2 axy kx/hbar tau}, {I axy qx/hbar tau, -2 ayz ky/hbar tau, Om, 2 ayx ky/hbar tau}, {I ayz qy/hbar tau, 2 axy kx/hbar tau, -2 ayx ky/hbar tau, Om}};
Kinv = Inverse[K] // FullSimplify;


Om = 1 - I om tau + I qv tau;
qv = qy hbar kF / m Sin[theta] + qx hbar kF / m Cos[theta];
kx = kF Cos[theta];
ky = kF Sin[theta];


(**Units all in terms of eV, ps, and Angstrom**)
hbar = 6.582*^-4;
kF = 0.015;
m = 0.00917 * 5.6856*^-8;
tau = 1; (** This is the scattering time from e-e, e-ph, or e-impurity interactions, NOT the spin lifetime **)
axy = -0.195165;
ayx = -0.155137;
ayz = 1.12069;
qx = 1*^-10; (** Set close to 0 **)
qy = 2 * Sqrt[ayx^2 + ayz^2] * m / hbar^2; (** Wavevector of PSH mode along the PST direction. **)


T = Pi * hbar / (Sqrt[ayx^2 + ayx^2] * kF)
Print["T"]
Print[T]

(** Debugging 
Export["K_s.csv", K];
Export["Kinv_s.csv", Kinv];
**)


$MaxExtraPrecision=100;
op=16; op1=13;


DD = Parallelize[Table[NIntegrate[Kinv[[i,j]], {theta, 0, 2 Pi},
WorkingPrecision->op,MaxRecursion->op1], {om, 1/(I taurange)}, {i,1,4},
{j,1,4}]] / (2 Pi);


(** The spin lifetime is tau when the real part of the determinant equals 0.
Imaginary part should always be 0. **)

Print["Determinant (Real)"]
Export["det_re.csv", Re[Table[Det[IdentityMatrix[4] - DD[[i]]], {i,Length[DD]}]]]
Print["Determinant (Imaginary)"]
Export["det_im.csv", Im[Table[Det[IdentityMatrix[4] - DD[[i]]], {i,Length[DD]}]]]
