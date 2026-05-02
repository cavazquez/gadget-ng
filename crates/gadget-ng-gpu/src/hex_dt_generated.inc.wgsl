// AUTO-GENERATED from hex_dt_patterns.rs — do not edit by hand

fn dt_x_0(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0 * rx * (15.0 * pow(r2, 2.0) - 70.0 * r2 * pow(rx, 2.0) + 63.0 * pow(rx, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_1(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    45.0 * ry
        * (4.0 * r2 * pow(rx, 2.0) - r2 * pow(ry, 2.0) - r2 * pow(rz, 2.0) - 12.0 * pow(rx, 4.0)
            + 9.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 9.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_2(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    45.0 * rz
        * (4.0 * r2 * pow(rx, 2.0) - r2 * pow(ry, 2.0) - r2 * pow(rz, 2.0) - 12.0 * pow(rx, 4.0)
            + 9.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 9.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_3(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rx
        * (3.0 * pow(r2, 2.0) - 7.0 * r2 * pow(rx, 2.0) - 21.0 * r2 * pow(ry, 2.0)
            + 63.0 * pow(rx, 2.0) * pow(ry, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_4(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    45.0 * rx * ry * rz * (4.0 * r2 - 18.0 * pow(rx, 2.0) + 3.0 * pow(ry, 2.0) + 3.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_5(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rx
        * (3.0 * pow(r2, 2.0) - 7.0 * r2 * pow(rx, 2.0) - 21.0 * r2 * pow(rz, 2.0)
            + 63.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_6(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * ry
        * (9.0 * r2 * pow(rx, 2.0) - 4.0 * r2 * pow(ry, 2.0) + 3.0 * r2 * pow(rz, 2.0)
            - 27.0 * pow(rx, 4.0)
            + 36.0 * pow(rx, 2.0) * pow(ry, 2.0)
            - 27.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_7(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rz
        * (3.0 * r2 * pow(rx, 2.0) - 6.0 * r2 * pow(ry, 2.0) + r2 * pow(rz, 2.0) - 9.0 * pow(rx, 4.0)
            + 54.0 * pow(rx, 2.0) * pow(ry, 2.0)
            - 9.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_8(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * ry
        * (3.0 * r2 * pow(rx, 2.0) + r2 * pow(ry, 2.0)
            - 6.0 * r2 * pow(rz, 2.0)
            - 9.0 * pow(rx, 4.0)
            - 9.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 54.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_9(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rz
        * (9.0 * r2 * pow(rx, 2.0) + 3.0 * r2 * pow(ry, 2.0)
            - 4.0 * r2 * pow(rz, 2.0)
            - 27.0 * pow(rx, 4.0)
            - 27.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 36.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_10(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * (pow(r2, 2.0) - 14.0 * r2 * pow(ry, 2.0) + 21.0 * pow(ry, 4.0)) / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_11(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * ry * rz * (2.0 * r2 - 9.0 * pow(rx, 2.0) + 12.0 * pow(ry, 2.0) - 9.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_12(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rx
        * (pow(r2, 2.0) - 7.0 * r2 * pow(ry, 2.0) - 7.0 * r2 * pow(rz, 2.0)
            + 63.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_13(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * ry * rz * (2.0 * r2 - 9.0 * pow(rx, 2.0) - 9.0 * pow(ry, 2.0) + 12.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_x_14(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * (pow(r2, 2.0) - 14.0 * r2 * pow(rz, 2.0) + 21.0 * pow(rz, 4.0)) / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_0(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * ry * (pow(r2, 2.0) - 14.0 * r2 * pow(rx, 2.0) + 21.0 * pow(rx, 4.0)) / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_1(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    15.0 * rx
        * (4.0 * r2 * pow(rx, 2.0)
            - 9.0 * r2 * pow(ry, 2.0)
            - 3.0 * r2 * pow(rz, 2.0)
            - 36.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 27.0 * pow(ry, 4.0)
            + 27.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_2(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * ry * rz * (2.0 * r2 + 12.0 * pow(rx, 2.0) - 9.0 * pow(ry, 2.0) - 9.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_3(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * ry
        * (3.0 * pow(r2, 2.0) - 21.0 * r2 * pow(rx, 2.0) - 7.0 * r2 * pow(ry, 2.0)
            + 63.0 * pow(rx, 2.0) * pow(ry, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_4(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    15.0 * rz
        * (6.0 * r2 * pow(rx, 2.0)
            - 3.0 * r2 * pow(ry, 2.0)
            - r2 * pow(rz, 2.0)
            - 54.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 9.0 * pow(ry, 4.0)
            + 9.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_5(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * ry
        * (pow(r2, 2.0) - 7.0 * r2 * pow(rx, 2.0) - 7.0 * r2 * pow(rz, 2.0)
            + 63.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_6(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0
        * rx
        * (r2 * pow(rx, 2.0) - 4.0 * r2 * pow(ry, 2.0) + r2 * pow(rz, 2.0)
            - 9.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 12.0 * pow(ry, 4.0)
            - 9.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_7(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    45.0 * rx * ry * rz * (4.0 * r2 + 3.0 * pow(rx, 2.0) - 18.0 * pow(ry, 2.0) + 3.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_8(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rx
        * (r2 * pow(rx, 2.0) + 3.0 * r2 * pow(ry, 2.0)
            - 6.0 * r2 * pow(rz, 2.0)
            - 9.0 * pow(rx, 2.0) * pow(ry, 2.0)
            - 9.0 * pow(ry, 4.0)
            + 54.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_9(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * ry * rz * (2.0 * r2 - 9.0 * pow(rx, 2.0) - 9.0 * pow(ry, 2.0) + 12.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_10(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0 * ry * (15.0 * pow(r2, 2.0) - 70.0 * r2 * pow(ry, 2.0) + 63.0 * pow(ry, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_11(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0
        * rz
        * (r2 * pow(rx, 2.0) - 4.0 * r2 * pow(ry, 2.0) + r2 * pow(rz, 2.0)
            - 9.0 * pow(rx, 2.0) * pow(ry, 2.0)
            + 12.0 * pow(ry, 4.0)
            - 9.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_12(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * ry
        * (3.0 * pow(r2, 2.0) - 7.0 * r2 * pow(ry, 2.0) - 21.0 * r2 * pow(rz, 2.0)
            + 63.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_13(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rz
        * (3.0 * r2 * pow(rx, 2.0) + 9.0 * r2 * pow(ry, 2.0)
            - 4.0 * r2 * pow(rz, 2.0)
            - 27.0 * pow(rx, 2.0) * pow(ry, 2.0)
            - 27.0 * pow(ry, 4.0)
            + 36.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_y_14(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * ry * (pow(r2, 2.0) - 14.0 * r2 * pow(rz, 2.0) + 21.0 * pow(rz, 4.0)) / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_0(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rz * (pow(r2, 2.0) - 14.0 * r2 * pow(rx, 2.0) + 21.0 * pow(rx, 4.0)) / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_1(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * ry * rz * (2.0 * r2 + 12.0 * pow(rx, 2.0) - 9.0 * pow(ry, 2.0) - 9.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_2(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    15.0 * rx
        * (4.0 * r2 * pow(rx, 2.0)
            - 3.0 * r2 * pow(ry, 2.0)
            - 9.0 * r2 * pow(rz, 2.0)
            - 36.0 * pow(rx, 2.0) * pow(rz, 2.0)
            + 27.0 * pow(ry, 2.0) * pow(rz, 2.0)
            + 27.0 * pow(rz, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_3(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rz
        * (pow(r2, 2.0) - 7.0 * r2 * pow(rx, 2.0) - 7.0 * r2 * pow(ry, 2.0)
            + 63.0 * pow(rx, 2.0) * pow(ry, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_4(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    15.0 * ry
        * (6.0 * r2 * pow(rx, 2.0)
            - r2 * pow(ry, 2.0)
            - 3.0 * r2 * pow(rz, 2.0)
            - 54.0 * pow(rx, 2.0) * pow(rz, 2.0)
            + 9.0 * pow(ry, 2.0) * pow(rz, 2.0)
            + 9.0 * pow(rz, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_5(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rz
        * (3.0 * pow(r2, 2.0) - 21.0 * r2 * pow(rx, 2.0) - 7.0 * r2 * pow(rz, 2.0)
            + 63.0 * pow(rx, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_6(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rx * ry * rz * (2.0 * r2 - 9.0 * pow(rx, 2.0) + 12.0 * pow(ry, 2.0) - 9.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_7(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rx
        * (r2 * pow(rx, 2.0) - 6.0 * r2 * pow(ry, 2.0) + 3.0 * r2 * pow(rz, 2.0)
            - 9.0 * pow(rx, 2.0) * pow(rz, 2.0)
            + 54.0 * pow(ry, 2.0) * pow(rz, 2.0)
            - 9.0 * pow(rz, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_8(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    45.0 * rx * ry * rz * (4.0 * r2 + 3.0 * pow(rx, 2.0) + 3.0 * pow(ry, 2.0) - 18.0 * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_9(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0
        * rx
        * (r2 * pow(rx, 2.0) + r2 * pow(ry, 2.0)
            - 4.0 * r2 * pow(rz, 2.0)
            - 9.0 * pow(rx, 2.0) * pow(rz, 2.0)
            - 9.0 * pow(ry, 2.0) * pow(rz, 2.0)
            + 12.0 * pow(rz, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_10(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0 * rz * (pow(r2, 2.0) - 14.0 * r2 * pow(ry, 2.0) + 21.0 * pow(ry, 4.0)) / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_11(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * ry
        * (3.0 * r2 * pow(rx, 2.0) - 4.0 * r2 * pow(ry, 2.0) + 9.0 * r2 * pow(rz, 2.0)
            - 27.0 * pow(rx, 2.0) * pow(rz, 2.0)
            + 36.0 * pow(ry, 2.0) * pow(rz, 2.0)
            - 27.0 * pow(rz, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_12(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0
        * rz
        * (3.0 * pow(r2, 2.0) - 21.0 * r2 * pow(ry, 2.0) - 7.0 * r2 * pow(rz, 2.0)
            + 63.0 * pow(ry, 2.0) * pow(rz, 2.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_13(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -45.0
        * ry
        * (r2 * pow(rx, 2.0) + r2 * pow(ry, 2.0)
            - 4.0 * r2 * pow(rz, 2.0)
            - 9.0 * pow(rx, 2.0) * pow(rz, 2.0)
            - 9.0 * pow(ry, 2.0) * pow(rz, 2.0)
            + 12.0 * pow(rz, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}

fn dt_z_14(rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    return (
    -15.0 * rz * (15.0 * pow(r2, 2.0) - 70.0 * r2 * pow(rz, 2.0) + 63.0 * pow(rz, 4.0))
        / pow(r2, 11.0 / 2.0)
    );
}
fn eval_dt_x(p: u32, rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    switch p {
        case 0u: { return dt_x_0(rx, ry, rz, r2); }
        case 1u: { return dt_x_1(rx, ry, rz, r2); }
        case 2u: { return dt_x_2(rx, ry, rz, r2); }
        case 3u: { return dt_x_3(rx, ry, rz, r2); }
        case 4u: { return dt_x_4(rx, ry, rz, r2); }
        case 5u: { return dt_x_5(rx, ry, rz, r2); }
        case 6u: { return dt_x_6(rx, ry, rz, r2); }
        case 7u: { return dt_x_7(rx, ry, rz, r2); }
        case 8u: { return dt_x_8(rx, ry, rz, r2); }
        case 9u: { return dt_x_9(rx, ry, rz, r2); }
        case 10u: { return dt_x_10(rx, ry, rz, r2); }
        case 11u: { return dt_x_11(rx, ry, rz, r2); }
        case 12u: { return dt_x_12(rx, ry, rz, r2); }
        case 13u: { return dt_x_13(rx, ry, rz, r2); }
        case 14u: { return dt_x_14(rx, ry, rz, r2); }
        default: { return 0.0; }
    }
}

fn eval_dt_y(p: u32, rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    switch p {
        case 0u: { return dt_y_0(rx, ry, rz, r2); }
        case 1u: { return dt_y_1(rx, ry, rz, r2); }
        case 2u: { return dt_y_2(rx, ry, rz, r2); }
        case 3u: { return dt_y_3(rx, ry, rz, r2); }
        case 4u: { return dt_y_4(rx, ry, rz, r2); }
        case 5u: { return dt_y_5(rx, ry, rz, r2); }
        case 6u: { return dt_y_6(rx, ry, rz, r2); }
        case 7u: { return dt_y_7(rx, ry, rz, r2); }
        case 8u: { return dt_y_8(rx, ry, rz, r2); }
        case 9u: { return dt_y_9(rx, ry, rz, r2); }
        case 10u: { return dt_y_10(rx, ry, rz, r2); }
        case 11u: { return dt_y_11(rx, ry, rz, r2); }
        case 12u: { return dt_y_12(rx, ry, rz, r2); }
        case 13u: { return dt_y_13(rx, ry, rz, r2); }
        case 14u: { return dt_y_14(rx, ry, rz, r2); }
        default: { return 0.0; }
    }
}

fn eval_dt_z(p: u32, rx: f32, ry: f32, rz: f32, r2: f32) -> f32 {
    switch p {
        case 0u: { return dt_z_0(rx, ry, rz, r2); }
        case 1u: { return dt_z_1(rx, ry, rz, r2); }
        case 2u: { return dt_z_2(rx, ry, rz, r2); }
        case 3u: { return dt_z_3(rx, ry, rz, r2); }
        case 4u: { return dt_z_4(rx, ry, rz, r2); }
        case 5u: { return dt_z_5(rx, ry, rz, r2); }
        case 6u: { return dt_z_6(rx, ry, rz, r2); }
        case 7u: { return dt_z_7(rx, ry, rz, r2); }
        case 8u: { return dt_z_8(rx, ry, rz, r2); }
        case 9u: { return dt_z_9(rx, ry, rz, r2); }
        case 10u: { return dt_z_10(rx, ry, rz, r2); }
        case 11u: { return dt_z_11(rx, ry, rz, r2); }
        case 12u: { return dt_z_12(rx, ry, rz, r2); }
        case 13u: { return dt_z_13(rx, ry, rz, r2); }
        case 14u: { return dt_z_14(rx, ry, rz, r2); }
        default: { return 0.0; }
    }
}
