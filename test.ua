P ← ÷⊙+⊃(×3ⁿ2|¯ⁿ2◌|××3⊙⋅∘)                # a b c ← (3ac - b^2)/(3a^2)
Q ← ÷⊙(++)⊃(×27ⁿ3|×2ⁿ3◌|¯×××9|××27ⁿ2⊙⋅⋅∘) # a b c d ← (2b^3 - 9abc + 27a^2d)/(27a^3)
S ← ¯÷×3                                  # a b ← -b/(3a)

R ← ×2√¯÷3⊙◌       # p q ← 2sqrt(-p/3)
A ← -:η°∿ ×3÷⊃R÷   # p q ← arccos(3(q/p)/R)
D ← ×⊃R(∿+η ÷3-τA) # p q ← R cos((A-2π)/3)

# solve ax^3 + bx^2 + cx + d = 0
# assumes 3 real roots - take the middle one
Cubic ← +D ⊃(P|Q|S)

a         ← 10
b         ← ¯30
Segments  ← 20
Framerate ← 24
Size      ← 4000_4000

Cb ← ×¯2+a b                  # -2(a+b)
Cc ← ++××2⊃(+b|+a|¯ⁿ2◌|×¯3ⁿ2) # y x ← 2(a+y)(b+y)-x^2-3y^2
Cd ← ×××¯2 a b                # y ← -2aby

Param     ← Cubic ⊃(1|Cb|Cc|Cd)
Normalise ← ÷-b a -b # (t-b)/(a-b)

Shade ← ◿2⌊ +⊓(×Segments|÷Framerate) # value time ← 0 or 1

fn = (
    ⇡×2Framerate
    Normalise Param °⊟-÷2⟜(°⍉ ≡(≡⊂:⇡4000)⇡4000)Size
    Shade¤
)
