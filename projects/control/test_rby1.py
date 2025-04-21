from dm_control import mujoco

xml_path = "models/rainbow_robotics_rby1/rby1.xml"
physics = mujoco.Physics.from_xml_path(xml_path)

# 1) 전체 정보
print("Total joints (njnt):", physics.model.njnt)  # ex) 29
print("Total qpos dims (nq):", physics.model.nq)   # ex) 35
print("Total dof (nv):", physics.model.nv)         # ex) 34 for a free joint

# 2) 각 joint별 이름/주소
for i in range(physics.model.njnt):
    joint_name = physics.model.id2name(i, 'joint')
    qpos_adr = physics.model.jnt_qposadr[i]
    dof_adr  = physics.model.jnt_dofadr[i]
    print(f"[Joint idx={i}] name={joint_name}, qpos_adr={qpos_adr}, dof_adr={dof_adr}")
