//! Templates PTX pour la génération de nombres aléatoires
//!
//! Ce module contient les kernels PTX pour la génération de nombres normalement
//! distribués et la conversion BF16.

/// Kernel de génération normale (Box-Muller)
///
/// Génère des nombres normalement distribués sur le GPU.
pub const NORMAL_GENERATION_KERNEL: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry normal_generation_kernel(
    .param .u64 output,
    .param .u64 seed,
    .param .u32 num_elements
) {
    .reg .u32 %r<10>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<10>;
    .reg .f64 %fd<10>;
    
    // Calculer l'index du thread
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    
    // Vérifier les limites
    ld.param.u32 %r5, [num_elements];
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra $END;
    
    // Charger la seed
    ld.param.u64 %rd1, [seed];
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd3, %rd2, 3;
    add.u64 %rd4, %rd1, %rd3;
    ld.u64 %rd5, [%rd4];
    
    // Générer u1
    mov.u64 %rd6, 6364136223846793005;
    mul.hi.u64 %rd7, %rd5, %rd6;
    add.u64 %rd8, %rd7, 1;
    cvt.u64.u32 %rd9, 33;
    shr.u64 %rd10, %rd8, %rd9;
    cvt.rn.f32.u64 %f1, %rd10;
    mov.f32 %f2, 2147483648.0;
    div.rn.f32 %f3, %f1, %f2;
    
    // Générer u2
    mul.hi.u64 %rd11, %rd8, %rd6;
    add.u64 %rd12, %rd11, 1;
    shr.u64 %rd13, %rd12, %rd9;
    cvt.rn.f32.u64 %f4, %rd13;
    div.rn.f32 %f5, %f4, %f2;
    
    // Box-Muller transform
    mov.f32 %f6, 1e-10;
    add.rn.f32 %f7, %f3, %f6;
    lg2.approx.f32 %f8, %f7;
    mov.f32 %f9, -2.0;
    mul.rn.f32 %f10, %f9, %f8;
    sqrt.rn.f32 %f11, %f10;
    
    // theta = 2π * u2
    mov.f32 %f12, 6.283185307179586;
    mul.rn.f32 %f13, %f12, %f5;
    
    // cos(theta)
    cos.approx.f32 %f14, %f13;
    mul.rn.f32 %f15, %f11, %f14;
    
    // Stocker le résultat
    ld.param.u64 %rd14, [output];
    shl.b64 %rd15, %rd2, 2;
    add.u64 %rd16, %rd14, %rd15;
    st.f32 [%rd16], %f15;
    
    // Mettre à jour la seed
    st.u64 [%rd4], %rd12;
    
    $END:
    ret;
}
"#;

/// Kernel de conversion F32 vers BF16
///
/// Convertit des float32 en bfloat16 pour optimiser la mémoire.
pub const BF16_CONVERSION_KERNEL: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry f32_to_bf16_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 num_elements
) {
    .reg .u32 %r<10>;
    .reg .u64 %rd<15>;
    .reg .f32 %f<5>;
    .reg .b32 %b<5>;
    
    // Calculer l'index du thread
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    
    // Vérifier les limites
    ld.param.u32 %r5, [num_elements];
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra $END;
    
    // Charger la valeur F32
    ld.param.u64 %rd1, [input];
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd4, %rd1, %rd3;
    ld.f32 %f1, [%rd4];
    
    // Convertir en bits
    mov.b32 %b1, %f1;
    
    // Extraire les 16 bits de poids fort (bfloat16)
    shr.b32 %b2, %b1, 16;
    cvt.u32.u32 %r6, %b2;
    
    // Stocker le résultat
    ld.param.u64 %rd5, [output];
    shl.b64 %rd6, %rd2, 1;
    add.u64 %rd7, %rd5, %rd6;
    st.u16 [%rd7], %r6;
    
    $END:
    ret;
}
"#;
