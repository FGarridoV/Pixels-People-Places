from UrbanTool.project import Project

p = Project('zones', utm = False, log = False, panoids=True)
p.generate_condensed_database()
