from django import template
from django.urls import reverse

register = template.Library()


# have to include icon code later
@register.inclusion_tag('theme/admin/menu_bar.html')
def show_admin_menu(user):
    main_menu = [
        {
            "name": "Users",
            "childs": [
                {
                    "name": "Applied Interviews",
                    "url": reverse("applied_interview_list"),
                },
                {
                    "name": "Decision Pending Interviews",
                    "url": reverse("decision_pending_interview_list"),
                },
                {
                    "name": "Finished Interview List",
                    "url": reverse("finished_interview_list"),
                }

            ]
        },

        {
            "name": "Project",
            "childs":
                [
                    {
                        "name": "Project List",
                        "url": reverse("project_list")
                    },
                    {
                        "name": "Admin Project List",
                        "url": reverse("admin_project_list")
                    }

                ]
        },
        {
            "name": "Domain",
            "childs":
                [
                    {
                        "name": "List Domain",
                        "url": reverse("list_domains"),
                        # "permission": 'kctusers.add_user'
                    },

                ]
        },
        {
            "name": "Inventory",
            "childs": [
                {
                    "name": "List Inventory",
                    "url": reverse("item_list"),
                },
                {
                    "name": "List Item Purchase",
                    "url": reverse("inventory_op_list"),
                },

            ]
        }

    ]

    filtered_menu = []
    user_permissions = user.get_all_permissions()
    for menu in main_menu:
        item = {'name': menu['name']}
        if "childs" in menu:
            item['childs'] = []
            for child in menu['childs']:
                if 'permission' in child:
                    if child['permission'] in user_permissions:
                        item["childs"].append(child)
                    else:
                        if isinstance(child['permission'], tuple):
                            for perm in child['permission']:
                                if perm in user_permissions:
                                    item["childs"].append(child)
                else:
                    item["childs"].append(child)
        filtered_menu.append(item)

    return {"menus": filtered_menu}
