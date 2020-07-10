! Copyright (C) 2017-2019 Matthieu Ancellin
! See LICENSE file at <https://github.com/mancellin/capytaine>
MODULE GREEN_WAVE

  USE CONSTANTS
  USE INITIALIZE_GREEN_WAVE
  USE GREEN_RANKINE, ONLY: COMPUTE_ASYMPTOTIC_RANKINE_SOURCE

  IMPLICIT NONE

  ! Dependancies between the functions of this module:
  ! (from top to bottom: "is called by")
  !
  !            LAGRANGE_POLYNOMIAL_INTERPOLATION
  !                          |
  !              COMPUTE_INTEGRALS_WRT_THETA       (COMPUTE_ASYMPTOTIC_RANKINE_SOURCE)
  !                        /   \                    /
  ! WAVE_PART_INFINITE_DEPTH   WAVE_PART_FINITE_DEPTH
  !                        \   /
  !                  (build_matrices.f90)
  !                          |
  !                    (python code)

CONTAINS

  ! =====================================================================

  SUBROUTINE LAGRANGE_POLYNOMIAL_INTERPOLATION &
    (dimless_r, dimless_Z,                                 &
     X_AXIS, Z_AXIS, TABULATION,               &
     D1, D2, Z1, Z2)
   ! Helper function used in the following subroutine to interpolate between the tabulated integrals.

    ! Inputs
    REAL(KIND=PRE),                        INTENT(IN) :: dimless_r, dimless_Z
    REAL(KIND=PRE), DIMENSION(3),          INTENT(IN) :: X_AXIS
    REAL(KIND=PRE), DIMENSION(3),          INTENT(IN) :: Z_AXIS
    REAL(KIND=PRE), DIMENSION(3, 3, 2, 2), INTENT(IN) :: TABULATION

    ! Output
    REAL(KIND=PRE), INTENT(OUT) :: D1, D2, Z1, Z2

    ! Local variable
    REAL(KIND=PRE), DIMENSION(3) :: XL, ZL

    XL(1) = PL2(X_AXIS(2), X_AXIS(3), X_AXIS(1), dimless_r)
    XL(2) = PL2(X_AXIS(3), X_AXIS(1), X_AXIS(2), dimless_r)
    XL(3) = PL2(X_AXIS(1), X_AXIS(2), X_AXIS(3), dimless_r)
    ZL(1) = PL2(Z_AXIS(2), Z_AXIS(3), Z_AXIS(1), dimless_Z)
    ZL(2) = PL2(Z_AXIS(3), Z_AXIS(1), Z_AXIS(2), dimless_Z)
    ZL(3) = PL2(Z_AXIS(1), Z_AXIS(2), Z_AXIS(3), dimless_Z)

    D1 = DOT_PRODUCT(XL, MATMUL(TABULATION(:, :, 1, 1), ZL))
    D2 = DOT_PRODUCT(XL, MATMUL(TABULATION(:, :, 2, 1), ZL))
    Z1 = DOT_PRODUCT(XL, MATMUL(TABULATION(:, :, 1, 2), ZL))
    Z2 = DOT_PRODUCT(XL, MATMUL(TABULATION(:, :, 2, 2), ZL))

  CONTAINS

    REAL(KIND=PRE) FUNCTION PL2(U1, U2, U3, XU)
      REAL(KIND=PRE) :: U1, U2, U3, XU
      PL2 = ((XU-U1)*(XU-U2))/((U3-U1)*(U3-U2))
      RETURN
    END FUNCTION

  END SUBROUTINE LAGRANGE_POLYNOMIAL_INTERPOLATION

  ! =====================================================================

  SUBROUTINE COMPUTE_INTEGRALS_WRT_THETA                         &
      (XI, XJ, wavenumber,                                       &
      tabulated_r_range, tabulated_Z_range, tabulated_integrals, &
      FS, VS)
    ! Compute the expression FS = 1/π Re[ ∫(J(ζ) - 1/ζ)dθ ] + i Re[ ∫(e^ζ)dθ ] and the integrals appearing in its gradient.
    ! For this, this function uses tabulated values of the following integrals:
    ! D1 = Re[ ∫(-i cosθ)(J(ζ) - 1/ζ)dθ ]
    ! D2 = Re[ ∫(-i cosθ)(e^ζ)dθ ]
#ifdef XIE_CORRECTION
    ! Z1 = Re[ ∫(J(ζ))dθ ]
#else
    ! Z1 = Re[ ∫(J(ζ) - 1/ζ)dθ ]
#endif
    ! Z2 = Re[ ∫(e^ζ)dθ ]
    ! (See theory manual for the definition of the symbols above.)

    ! Inputs
    REAL(KIND=PRE), DIMENSION(3),             INTENT(IN) :: XI, XJ
    REAL(KIND=PRE),                           INTENT(IN) :: wavenumber

    ! Tabulated data
    REAL(KIND=PRE), DIMENSION(328),           INTENT(IN) :: tabulated_r_range
    REAL(KIND=PRE), DIMENSION(46),            INTENT(IN) :: tabulated_Z_range
    REAL(KIND=PRE), DIMENSION(328, 46, 2, 2), INTENT(IN) :: tabulated_integrals

    ! Outputs
    COMPLEX(KIND=PRE),                        INTENT(OUT) :: FS  ! the integral
    COMPLEX(KIND=PRE), DIMENSION(3),          INTENT(OUT) :: VS  ! its gradient

    ! Local variables
    INTEGER        :: KI, KJ
    REAL(KIND=PRE) :: r, dimless_r, Z, dimless_Z, R1, dimless_R1
    REAL(KIND=PRE) :: sin_kr, cos_kr, expz_sqr
    REAL(KIND=PRE) :: D1, D2, Z1, Z2

    r = NORM2(XI(1:2) - XJ(1:2))
    dimless_r = wavenumber*r

    Z = XI(3) + XJ(3)
    dimless_Z = wavenumber*Z

    R1 = SQRT(r**2 + Z**2)
    dimless_R1 = wavenumber*R1

    IF ((R1 < 1e-5) .OR. (wavenumber == 0)) THEN
      PRINT*, "Error: Impossible to compute the wave part of the Green function (division by zero)."
      ERROR STOP
    ENDIF

    !=====================================================================================
    ! Evaluate the elementary integrals PDnX and PDnZ depending on dimless_Z and dimless_r
    !=====================================================================================
    IF (((MINVAL(tabulated_Z_range) < dimless_Z) .AND. (dimless_Z < MAXVAL(tabulated_Z_range))) .AND. &
        ((MINVAL(tabulated_r_range) <= dimless_r) .AND. (dimless_r < MAXVAL(tabulated_r_range)))) THEN
        ! Within the range of tabulated data
        ! Note that MINVAL(tabulated_r_range) == 0, so one of the conditions is not actually useful.

        ! Get the nearest point in the tabulation
        IF (dimless_r < 1) THEN
          KI = INT(5*(LOG10(dimless_r+1e-20)+6)+1)
        ELSE
          KI = INT(3*dimless_r+28)
        ENDIF
        KI = MAX(MIN(KI, 327), 2)

        IF (dimless_Z < -1e-2) THEN
          KJ = INT(8*(LOG10(-dimless_Z)+4.5))
        ELSE
          KJ = INT(5*(LOG10(-dimless_Z)+6))
        ENDIF
        KJ = MAX(MIN(KJ, 45), 2)

        ! Interpolate near this point to get the actual value
        CALL LAGRANGE_POLYNOMIAL_INTERPOLATION                           &
             (dimless_r, dimless_Z,                                      &
             tabulated_r_range(KI-1:KI+1), tabulated_Z_range(KJ-1:KJ+1), &
             tabulated_integrals(KI-1:KI+1, KJ-1:KJ+1, :, :),            &
             D1, D2, Z1, Z2)

      ELSE  ! MAXVAL(tabulated_r_range) < dimless_r
        ! Asymptotic expression for (horizontally) distant panels

        expz_sqr = EXP(dimless_Z) * SQRT(2*PI/dimless_r)
        cos_kr  = COS(dimless_r - PI/4)
        sin_kr  = SIN(dimless_r - PI/4)

        D1 = PI*(expz_sqr*(cos_kr - sin_kr/(2*dimless_r)) - dimless_r/dimless_R1**3)
        D2 =     expz_sqr*(sin_kr + cos_kr/(2*dimless_r))
#ifdef XIE_CORRECTION
        Z1 = PI*(-expz_sqr*sin_kr + dimless_Z/dimless_R1**3 - ONE/dimless_R1)
#else
        Z1 = PI*(-expz_sqr*sin_kr + dimless_Z/dimless_R1**3)
#endif
        Z2 =     expz_sqr*cos_kr
      ENDIF

      !================================================
      ! Add the elementary integrals to build FS and VS
      !================================================

#ifdef XIE_CORRECTION
      FS    = CMPLX(Z1/PI + ONE/dimless_R1, Z2, KIND=PRE)
      VS(3) = CMPLX(Z1/PI + ONE/dimless_R1, Z2, KIND=PRE)
#else
      FS    = CMPLX(Z1/PI, Z2, KIND=PRE)
      VS(3) = CMPLX(Z1/PI, Z2, KIND=PRE)
#endif
      VS(1) = (XJ(1) - XI(1))/r * CMPLX(D1/PI, D2, KIND=PRE)
      VS(2) = (XJ(2) - XI(2))/r * CMPLX(D1/PI, D2, KIND=PRE)

      IF (r < REAL(1e-5, KIND=PRE)) THEN
        ! Limit case r ~ 0 ?
        VS(1:2) = CMPLX(0.0, 0.0, KIND=PRE)
      END IF

    RETURN
  END SUBROUTINE COMPUTE_INTEGRALS_WRT_THETA

  ! =========================

  SUBROUTINE WAVE_PART_INFINITE_DEPTH &
      (wavenumber, X0I, X0J,          &
      X_AXIS, Z_AXIS, TABULATION,     &
      SP, VSP)
    ! Compute the wave part of the Green function in the infinite depth case.
    ! This is mostly the integral computed by the subroutine above.

    ! Inputs
    REAL(KIND=PRE),                           INTENT(IN)  :: wavenumber
    REAL(KIND=PRE), DIMENSION(3),             INTENT(IN)  :: X0I   ! Coordinates of the source point
    REAL(KIND=PRE), DIMENSION(3),             INTENT(IN)  :: X0J   ! Coordinates of the center of the integration panel

    ! Tabulated data
    REAL(KIND=PRE), DIMENSION(328),           INTENT(IN) :: X_AXIS
    REAL(KIND=PRE), DIMENSION(46),            INTENT(IN) :: Z_AXIS
    REAL(KIND=PRE), DIMENSION(328, 46, 2, 2), INTENT(IN) :: TABULATION

    ! Outputs
    COMPLEX(KIND=PRE),               INTENT(OUT) :: SP  ! Integral of the Green function over the panel.
    COMPLEX(KIND=PRE), DIMENSION(3), INTENT(OUT) :: VSP ! Gradient of the integral of the Green function with respect to X0I.

    ! Local variables
    REAL(KIND=PRE), DIMENSION(3) :: XJ_REFLECTION

    ! The integrals
    CALL COMPUTE_INTEGRALS_WRT_THETA(X0I, X0J, wavenumber, X_AXIS, Z_AXIS, TABULATION, SP, VSP(:))
    SP  = 2*wavenumber*SP
    VSP = 2*wavenumber**2*VSP

    ! Only one singularity is missing in the derivative
    XJ_REFLECTION(1:2) = X0J(1:2)
    XJ_REFLECTION(3) = - X0J(3)
    VSP = VSP - 2*(X0I - XJ_REFLECTION)/(NORM2(X0I-XJ_REFLECTION)**3)

    RETURN
  END SUBROUTINE WAVE_PART_INFINITE_DEPTH

  ! ======================

  SUBROUTINE WAVE_PART_FINITE_DEPTH &
      (wavenumber, X0I, X0J, depth, &
      X_AXIS, Z_AXIS, TABULATION,                  &
      NEXP, AMBDA, AR,              &
      SP, VSP_SYM, VSP_ANTISYM)
    ! Compute the frequency-dependent part of the Green function in the finite depth case.

    ! Inputs
    REAL(KIND=PRE),                           INTENT(IN) :: wavenumber, depth
    REAL(KIND=PRE), DIMENSION(3),             INTENT(IN) :: X0I  ! Coordinates of the source point
    REAL(KIND=PRE), DIMENSION(3),             INTENT(IN) :: X0J  ! Coordinates of the center of the integration panel

    REAL(KIND=PRE), DIMENSION(328),           INTENT(IN) :: X_AXIS
    REAL(KIND=PRE), DIMENSION(46),            INTENT(IN) :: Z_AXIS
    REAL(KIND=PRE), DIMENSION(328, 46, 2, 2), INTENT(IN) :: TABULATION

    INTEGER,                                  INTENT(IN) :: NEXP
    REAL(KIND=PRE), DIMENSION(NEXP),          INTENT(IN) :: AMBDA, AR

    ! Outputs
    COMPLEX(KIND=PRE),               INTENT(OUT) :: SP  ! Integral of the Green function over the panel.
    COMPLEX(KIND=PRE), DIMENSION(3), INTENT(OUT) :: VSP_SYM, VSP_ANTISYM ! Gradient of the integral of the Green function with respect to X0I.

    ! Local variables
    INTEGER                              :: KE
    REAL(KIND=PRE)                       :: AMH, AKH, A
    REAL(KIND=PRE)                       :: AQT, R
    REAL(KIND=PRE),    DIMENSION(3)      :: XI, XJ
    REAL(KIND=PRE),    DIMENSION(4)      :: FTS, PSR
    REAL(KIND=PRE),    DIMENSION(3, 4)   :: VTS
    COMPLEX(KIND=PRE), DIMENSION(4)      :: FS
    COMPLEX(KIND=PRE), DIMENSION(3, 4)   :: VS

    !========================================
    ! Part 1: Solve 4 infinite depth problems
    !========================================

    XI(:) = X0I(:)
    XJ(:) = X0J(:)

    ! Distance in xOy plane
    R = NORM2(XI(1:2) - XJ(1:2))

    ! 1.a First infinite depth problem
    CALL COMPUTE_INTEGRALS_WRT_THETA(XI(:), XJ(:), wavenumber, X_AXIS, Z_AXIS, TABULATION, FS(1), VS(:, 1))

    PSR(1) = ONE/(wavenumber*SQRT(R**2+(XI(3)+XJ(3))**2))

    ! 1.b Shift and reflect XI and compute another value of the Green function
    XI(3) = -X0I(3) - 2*depth
    XJ(3) =  X0J(3)
    CALL COMPUTE_INTEGRALS_WRT_THETA(XI(:), XJ(:), wavenumber, X_AXIS, Z_AXIS, TABULATION, FS(2), VS(:, 2))
    VS(3, 2) = -VS(3, 2) ! Reflection of the output vector

    PSR(2) = ONE/(wavenumber*SQRT(R**2+(XI(3)+XJ(3))**2))

    ! 1.c Shift and reflect XJ and compute another value of the Green function
    XI(3) =  X0I(3)
    XJ(3) = -X0J(3) - 2*depth
    CALL COMPUTE_INTEGRALS_WRT_THETA(XI(:), XJ(:), wavenumber, X_AXIS, Z_AXIS, TABULATION, FS(3), VS(:, 3))

    PSR(3) = ONE/(wavenumber*SQRT(R**2+(XI(3)+XJ(3))**2))

    ! 1.d Shift and reflect both XI and XJ and compute another value of the Green function
    XI(3) = -X0I(3) - 2*depth
    XJ(3) = -X0J(3) - 2*depth
    CALL COMPUTE_INTEGRALS_WRT_THETA(XI(:), XJ(:), wavenumber, X_AXIS, Z_AXIS, TABULATION, FS(4), VS(:, 4))
    VS(3, 4) = -VS(3, 4) ! Reflection of the output vector

    PSR(4) = ONE/(wavenumber*SQRT(R**2+(XI(3)+XJ(3))**2))

    ! Add up the results of the four problems
    SP               = SUM(FS(1:4)) - SUM(PSR(1:4))
    VSP_SYM(1:3)     = VS(1:3, 1) + VS(1:3, 4)
    VSP_ANTISYM(1:3) = VS(1:3, 2) + VS(1:3, 3)

    ! Multiply by some coefficients
    AMH  = wavenumber*depth
    AKH  = AMH*TANH(AMH)
    A    = (AMH+AKH)**2/(2*depth*(AMH**2-AKH**2+AKH))

    SP          = A*SP
    VSP_ANTISYM = A*wavenumber*VSP_ANTISYM
    VSP_SYM     = A*wavenumber*VSP_SYM

    !=====================================================
    ! Part 2: Integrate (NEXP+1)×4 terms of the form 1/MM'
    !=====================================================

    DO KE = 1, NEXP
      XI(:) = X0I(:)

      ! 2.a Shift observation point and compute integral
      XI(3) =  X0I(3) + depth*AMBDA(KE) - 2*depth
      CALL COMPUTE_ASYMPTOTIC_RANKINE_SOURCE(XI(:), X0J(:), ONE, FTS(1), VTS(:, 1))

      ! 2.b Shift and reflect observation point and compute integral
      XI(3) = -X0I(3) - depth*AMBDA(KE)
      CALL COMPUTE_ASYMPTOTIC_RANKINE_SOURCE(XI(:), X0J(:), ONE, FTS(2), VTS(:, 2))
      VTS(3, 2) = -VTS(3, 2) ! Reflection of the output vector

      ! 2.c Shift and reflect observation point and compute integral
      XI(3) = -X0I(3) + depth*AMBDA(KE) - 4*depth
      CALL COMPUTE_ASYMPTOTIC_RANKINE_SOURCE(XI(:), X0J(:), ONE, FTS(3), VTS(:, 3))
      VTS(3, 3) = -VTS(3, 3) ! Reflection of the output vector

      ! 2.d Shift observation point and compute integral
      XI(3) =  X0I(3) - depth*AMBDA(KE) + 2*depth
      CALL COMPUTE_ASYMPTOTIC_RANKINE_SOURCE(XI(:), X0J(:), ONE, FTS(4), VTS(:, 4))

      AQT = AR(KE)/2

      ! Add all the contributions
      SP               = SP               + AQT*SUM(FTS(1:4))
      VSP_ANTISYM(1:3) = VSP_ANTISYM(1:3) + AQT*(VTS(1:3, 1) + VTS(1:3, 4))
      VSP_SYM(1:3)     = VSP_SYM(1:3)     + AQT*(VTS(1:3, 2) + VTS(1:3, 3))

    END DO

    RETURN
  END SUBROUTINE

END MODULE GREEN_WAVE
